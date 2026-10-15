# encoding: utf-8
try:
    from .classify import load_cls, label_chars, load_cnn_predictor
    from .fast_utils import fnormalize, ftrim, gausslogprob, fadd_padding
    from .feature_extraction import normalize_and_extract_features
    from .config_manager import default_config
except ImportError:
    from classify import load_cls, label_chars, load_cnn_predictor
    from fast_utils import fnormalize, ftrim, gausslogprob, fadd_padding
    from feature_extraction import normalize_and_extract_features
    from config_manager import default_config
import cv2 as cv
from collections import namedtuple
from numpy import floor, uint8, ones, argmax, hstack, mean, std, \
    ceil, load
import numpy as np
from random import gauss
import os

# Small parameter bundles that keep the segmentation methods within the
# parameter-count limit. Dist = a (mean, std) width distribution; FinalData =
# the three parallel per-page accumulators (boxes / contour-indices / box stats);
# ExpSegContext = the invariant geometry+stats a (chars, z) experimental break
# candidate is evaluated against.
Dist = namedtuple("Dist", "mean std")
FinalData = namedtuple("FinalData", "boxes indices box_info")
ExpSegContext = namedtuple("ExpSegContext", "char_mean_int vsum x y scale_l final_box_info")
try:
    from .page_elements2 import PageElements
    from .transitions import horizontal_transitions
    from .utils import local_file, check_for_overlap
    from .viterbi_cython import viterbi_cython
    from .combine_boxes import CombineBoxesForPage
    from .segment_experimental import ExperimentalSegmenterMixin
except ImportError:
    from page_elements2 import PageElements
    from transitions import horizontal_transitions
    from utils import local_file, check_for_overlap
    from viterbi_cython import viterbi_cython
    from combine_boxes import CombineBoxesForPage
    from segment_experimental import ExperimentalSegmenterMixin


def _dbg(msg):
    """Print a debug message when debug_output is enabled (branch lives here
    so hot-path call sites stay decision-free)."""
    if default_config.get('debug_output', False):
        print(msg)


def build_vector_set(seg, process_line, combine_naros):
    """Shared body of construct_vector_set_{stochastic,experimental}: build the
    page's CombineBoxesForPage, reset the per-line vector/box lists, run the
    path-specific ``process_line(l, final_boxes, final_indices, final_box_info,
    scales)`` for each line, and (unless the page produced no vectors) run the
    path-specific ``combine_naros()`` when detect_o is on."""
    final_box_info = CombineBoxesForPage(seg.line_info)
    seg.final_box_info = final_box_info
    final_boxes = final_box_info.final_boxes
    final_indices = final_box_info.final_indices
    scales = final_box_info.transitions

    seg.vectors = [[] for i in range(seg.line_info.k)]
    seg.new_boxes = [[] for i in range(seg.line_info.k)]

    for l in range(len(final_indices)):  # for each line
        process_line(l, final_boxes, final_indices, final_box_info, scales)

    if not any(seg.vectors):
        _dbg('no vectors')
        return
    if seg.line_info.shapes.detect_o:
        combine_naros()


def _ink_projection(letter):
    # Ensure we're counting ink pixels (handle both conventions)
    if letter.max() <= 1:
        ink = (1 - letter).astype(np.float32)  # invert: ink pixels become 1
    else:
        ink = (letter < 128).astype(np.float32)
    # Vertical projection: sum ink pixels per column
    return np.sum(ink, axis=0)


def _valley_candidates(projection, margin, threshold):
    # Find local minima below threshold in search region
    candidates = []
    for x in range(margin, len(projection) - margin):
        if projection[x] <= threshold:
            if projection[x] <= projection[x - 1] and projection[x] <= projection[x + 1]:
                candidates.append((x, float(projection[x])))
    return candidates


def _merge_valley_candidates(candidates, merge_dist):
    # Merge nearby candidates (keep the one with lowest ink)
    merged = []
    current_group = [candidates[0]]

    for i in range(1, len(candidates)):
        if candidates[i][0] - current_group[-1][0] <= merge_dist:
            current_group.append(candidates[i])
        else:
            best = min(current_group, key=lambda c: c[1])
            merged.append(best[0])
            current_group = [candidates[i]]

    best = min(current_group, key=lambda c: c[1])
    merged.append(best[0])

    return merged


def find_projection_split_points(letter, char_mean):
    """Find split points in a wide character blob using vertical ink projection.

    Counts ink pixels per column and finds valleys (low-ink columns) that
    indicate boundaries between connected characters.

    Args:
        letter: Binary image array (ink=1, background=0 or ink=0, background=1)
        char_mean: Mean character width for the current line

    Returns:
        List of x-coordinates where the blob should be split, or empty list.
    """
    h, w = letter.shape
    if w < 2 * char_mean * 0.3:
        return []

    projection = _ink_projection(letter)

    if projection.mean() == 0:
        return []

    # Search the middle region (skip 15% edges to avoid cutting into characters)
    margin = max(3, int(w * 0.15))

    threshold = projection.mean() * 0.3

    candidates = _valley_candidates(projection, margin, threshold)

    if not candidates:
        return []

    merge_dist = max(3, int(char_mean * 0.15))
    return _merge_valley_candidates(candidates, merge_dist)


# --- CNN predictor with sklearn fallback ---
predictor, use_cnn = load_cnn_predictor("[segment.py]")
if not use_cnn:
    cls = load_cls('logistic-cls')

## commonly called functions
GaussianBlur = cv.GaussianBlur
if use_cnn:
    predict_log_proba = predictor.predict_log_proba
else:
    predict_log_proba = cls.predict_log_proba
boundingRect = cv.boundingRect
char_gaussians = PageElements.char_gaussians


def _normalize_to_32x32(arr):
    """Normalize image to 32x32 for CNN input."""
    buf = np.ones((32, 32), dtype=np.uint8)
    fnormalize(arr, buf)
    return buf

trans_p = load(open(local_file('stack_bigram_logprob32.npz'),'rb'))
trans_p = trans_p[trans_p.files[0]].transpose()
start_p = load(open(local_file('stack_start_logprob32.npz'),'rb'))
start_p = start_p[start_p.files[0]]
n_states = trans_p.shape[0]


def combine_many_boxes(bxs):
    '''Return the largest bounding box using max height and width from all boxes

    bxs is a list of boxes

    returns (x,y,w,h) of the new box
    '''

    if not bxs:
        raise ValueError('No boxes specified')

    new_top = min([b[1] for b in bxs])
    new_bottom = max([b[1]+b[3] for b in bxs])
    new_left = min([b[0] for b in bxs])
    new_right = max([b[0]+b[2] for b in bxs])
    return (new_left, new_top, new_right-new_left, new_bottom-new_top)

def image_to_32x16_vector(im):
    a = ones((32,16), dtype=uint8)
    h,w = im.shape
    s = min(32.0/h, 16.0/w)
    b = cv.resize(im, (0,0), fy=s, fx=s, interpolation=cv.INTER_AREA)
    a[0:b.shape[0], 0:b.shape[1]] = b
    return a.flatten()


def normalized_scale(arr):
    crss_denom = float(max(horizontal_transitions(arr)))
    if crss_denom > 0:
        crossings_val = (arr.shape[1])/crss_denom
        scale =  35.0 / crossings_val
    else:
        scale = 1.0

    return scale



class Segmenter(ExperimentalSegmenterMixin):
    def __init__(self, line_info, break_resolution = 6, draw_outlines=True):
        self.line_info = line_info
        self.draw_outlines = draw_outlines
        self.break_window_resolution = break_resolution
        self.breakwidth = line_info.shapes.conf['break_width']
        self.cached_features = line_info.shapes.cached_features

        if line_info.shapes.conf['segmenter'] == 'experimental':
            self.construct_vector_set_experimental()
        elif line_info.shapes.conf['segmenter'] == 'stochastic':
            self.construct_vector_set_stochastic()


    def _min_variance_breakwidth(self):
        widths = self.final_box_info.widths
        char_mean = self.final_box_info.char_mean
        char_std = self.final_box_info.char_std
        ws = [1.75, 2.5, 2.75, 3.0, 3.6, 4.0, 8.0]

        new_widths = [[] for i in range(len(ws))]
        for wd in widths:
            for i, w in enumerate(ws):
                if wd >= char_mean + w*char_std :

                    splits = int(floor(float(wd)/(char_mean-char_std)))
                    for u in range(splits):
                        new_widths[i].append(char_mean)
                    else:
                        new_widths[i].append(wd)
                else:
                    new_widths[i].append(wd)

        best_var_arg = np.argmin([np.var(wnews) for wnews in new_widths])
        return ws[best_var_arg]



    def _apply_naro_overlap(self, s, nnbox, line_num, oo_scale_l):
        """If this sub-image overlaps a detected naro (ོ), grow it to include the
        naro contour. Returns the (possibly enlarged) (s, nnbox)."""
        naro = self.line_info.check_naro_overlap(line_num, nnbox)
        if naro is False:
            return s, nnbox
        naro_box = self.line_info.get_box(naro)
        nnbox = combine_many_boxes([nnbox, naro_box])
        ss = cv.resize(s, dsize=(0,0), fx=oo_scale_l, fy=oo_scale_l)
        ss = np.vstack((ones((nnbox[3]-ss.shape[0], ss.shape[1]), dtype=ss.dtype), ss))
        ss = hstack((ss, ones((ss.shape[0], nnbox[2] - ss.shape[1]), dtype=ss.dtype)))
        cv.drawContours(ss, [self.line_info.get_contour(naro)], -1, 0, thickness=-1, offset=(-naro_box[0], -naro_box[1]))
        return ss, nnbox

    @staticmethod
    def _fill_small_contours(s, padding_amount):
        # OpenCV version-agnostic findContours (3.x returns 3-tuple, 4.x a 2-tuple)
        contours_result = cv.findContours(s.copy(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        ctrs, hier = contours_result[-2:]
        for k, b in enumerate(map(boundingRect, ctrs)):
            if (b[2] < 23 or b[3] < 23) and hier[0][k][3] == 0:
                s[b[1]-1:b[1]+b[3]+1, b[0]-1:b[0]+b[2]+1] = 1
        return s[padding_amount:-padding_amount, padding_amount:-padding_amount]

    def _segment_widths_pass(self, letter, widths, box_xy, oo_scale_l, line_num,
                             dist, padding_amount):
        """One random-width partition pass: cut `letter` into len(widths) slices,
        classify each. Returns (vecs, boxes, wdthprobs); stops early (as the
        original `break`) if a slice trims to an empty image."""
        cur_mean, cur_std = dist
        x, y = box_xy
        chars = len(widths)
        prev = 0
        vecs = []
        boxes = []
        wdthprobs = 0
        for i, val in enumerate(widths):
            end = letter.shape[1] if i == chars - 1 else prev+val
            wdthprobs += gausslogprob(cur_mean, cur_std, end-prev)

            s = fadd_padding(letter[:, int(prev):int(end)], padding_amount)
            s = self._fill_small_contours(s, padding_amount)
            s, ofst = ftrim(s, new_offset=True)

            if 0 in s.shape:
                break
            nnbox = [x+(prev + ofst['left'])*oo_scale_l, y + (ofst['top']*oo_scale_l), s.shape[1]*oo_scale_l, s.shape[0]*oo_scale_l]
            if line_num is not None:
                s, nnbox = self._apply_naro_overlap(s, nnbox, line_num, oo_scale_l)
            vecs.append(_normalize_to_32x32(s) if use_cnn else normalize_and_extract_features(s))
            boxes.append(nnbox)
            prev += val
        return vecs, boxes, wdthprobs

    def _sample_multichar(self, chars, letter, box_xy, oo_scale_l, line_num, cur_mean, cur_std):
        """Multi-character break: 15 random-width sampling passes, keep the best
        Viterbi score. Returns (best_prob, best_prd, best_boxes)."""
        letter = cv.dilate(letter.copy(), None, iterations=1)
        padding_amount = 3
        best_prob = -np.inf
        best_prd = None
        best_boxes = None
        for n in range(15):
            widths = [gauss(cur_mean, cur_std) for _ in range(chars)]
            vecs, boxes, wdthprobs = self._segment_widths_pass(
                letter, widths, box_xy, oo_scale_l, line_num, Dist(cur_mean, cur_std), padding_amount)
            if not vecs:
                continue
            xn = len(vecs)
            if use_cnn:
                vecs = np.array(vecs)  # (xn, 32, 32)
            else:
                vecs = np.array(vecs).reshape(xn, 346)  # 346 is len(vecs[0])

            probs = predict_log_proba(vecs).astype(np.float32)
            if n % 10 == 0 and n != 0:
                cur_mean = self.final_box_info.char_mean*(.97-(3*n/1000.0))

            prob, prds = viterbi_cython(xn, n_states, start_p, trans_p, probs)
            prob = prob + wdthprobs
            if prob > best_prob:
                best_prob = prob
                best_prd = prds
                best_boxes = boxes
        return best_prob, best_prd, best_boxes

    def _classify_single(self, letter, letter_box, oo_scale_l, cur_mean, cur_std):
        """Single-character case: classify the whole letter as one glyph."""
        best_boxes = [letter_box]
        if use_cnn:
            probs = predict_log_proba(_normalize_to_32x32(letter))
        else:
            features = normalize_and_extract_features(letter, debug_coordinates=letter_box)
            probs = predict_log_proba(features.reshape(1, -1))
        amx = probs[0].argmax()
        try:
            startprob = start_p[amx]
        except IndexError:
            startprob = 1e-10
        best_prob = probs[0][amx] + gausslogprob(cur_mean, cur_std, letter_box[2]/oo_scale_l) + startprob
        return best_prob, [amx], best_boxes

    def _sample_widths_method(self, chars, letter, letter_box, oo_scale_l, line_num=None):
        x, y, w, h = letter_box
        cur_mean = self.final_box_info.char_mean*.97
        cur_std = .295*self.final_box_info.char_std
        _dbg(f"[SEGMENT DEBUG] Breaking {chars} chars: cur_mean={cur_mean:.1f}, cur_std={cur_std:.1f}, box_width={w}")

        if chars > 1:
            best_prob, best_prd, best_boxes = self._sample_multichar(
                chars, letter, (x, y), oo_scale_l, line_num, cur_mean, cur_std)
        else:
            best_prob, best_prd, best_boxes = self._classify_single(
                letter, letter_box, oo_scale_l, cur_mean, cur_std)

        final_prob = best_prob
        res = []
        # best_prd is None only when no multichar sampling pass beat -inf (all passes
        # produced empty vecs) — degrade to no boxes rather than crashing (the original
        # raised on `enumerate(None)`).
        if best_prd is not None:
            for i, val in enumerate(best_prd):
                best_boxes[i] = [int(np.round(k)) for k in best_boxes[i]]
                best_boxes[i].extend([float(np.exp(final_prob)), label_chars[val]])
                res.append(best_boxes[i])

        return (final_prob, res)


    def _detach_tsek(self):
        # Unimplemented stub (no-op). The single call site self._detach_tsek()
        # passes no args, so the signature takes none — the future `letter`-based
        # implementation is sketched in the comments below.
        # 1. check if detach makes sense: i.e. will chopping off end result in
        # something that looks and acts like a tsek, in size and position
        # 2. isolate the tsek-part, create a new bounding box for it
        # update the parent box with new dimensions

        #         tsek_part = letter[:, letter.shape[1]-tsek_mean:]


        pass

    def construct_vector_set_stochastic(self):
        # separate attached tsek
        # note this may note go here exactly, but somewhere in this function
        if self.line_info.shapes.conf.get('detach_tsek'):
            self._detach_tsek()

        build_vector_set(self, self._process_stochastic_line, self._combine_naros_stochastic)

    def _process_stochastic_line(self, l, final_boxes, final_indices, final_box_info, scales):
        try:
            scale_l = scales[l]
            oo_scale_l = 1.0/scale_l
        except:
            print(('ERROR AT ', l, len(scales)))
            raise
        try:
            lb = list(range(len(final_indices[l])))
        except IndexError:
            return

        final = FinalData(final_boxes, final_indices, final_box_info)
        for i in lb: # for each line box
            self._process_stochastic_box(l, i, final, scale_l, oo_scale_l, self.breakwidth)

    def _draw_char_letter(self, box, lindices):
        """Rasterize the contours of one line box into a fresh letter array."""
        x, y, w, h = box
        letter = ones((h, w), dtype=uint8)
        for k in lindices:
            if not isinstance(k, str):
                letter = self.line_info.shapes.draw_contour_and_children(k, char_arr=letter, offset=(-x, -y))
            else:
                cv.drawContours(letter, [self.line_info.get_contour(k)], -1, 0, thickness=-1, offset=(-x, -y))
        return letter

    @staticmethod
    def _conservative_breakpoint(final_box_info, breakwidth, letter):
        # Scale-normalized threshold from reference char sizes (not raw measurements),
        # deliberately conservative to avoid over-segmenting legitimate wide letters.
        reference_char_mean = 14.23  # From paragraph2.png reference
        reference_char_std = 3.0     # Typical standard deviation
        scale_factor = getattr(final_box_info, 'global_scale_factor', final_box_info.char_mean / reference_char_mean)
        reference_breakpoint = reference_char_mean + (breakwidth + 1.2) * reference_char_std
        bp = reference_breakpoint * scale_factor
        if default_config.get('debug_output', False):
            print(f"[SEGMENT DEBUG] Character analysis: width={letter.shape[1]:.1f}, conservative_breakpoint={bp:.1f}")
            print(f"[SEGMENT DEBUG] Threshold calc: char_mean={final_box_info.char_mean:.3f}, char_std={final_box_info.char_std:.3f}, scale_factor={scale_factor:.3f}")
        return bp

    def _projection_split_box(self, l, i, letter, box, split_points):
        """Cut a wide blob at the projection split points and push each sub-image
        through the classifier. Returns True (caller skips normal processing)."""
        x, y, w, h = box
        _dbg(f"[PROJECTION SPLIT] Found {len(split_points)} split point(s) at columns {split_points} for {letter.shape[1]}px blob")
        boundaries = [0] + split_points + [letter.shape[1]]
        for seg_idx in range(len(boundaries) - 1):
            seg_left = boundaries[seg_idx]
            seg_right = boundaries[seg_idx + 1]
            sub_letter = letter[:, seg_left:seg_right]

            # Skip empty or trivially small sub-images
            if sub_letter.shape[1] < 3 or np.sum(sub_letter < 0.5) < 5:
                _dbg(f"[PROJECTION SPLIT] Skipping empty sub-image {seg_idx}: {sub_letter.shape}")
                continue

            seg_box = [x + seg_left, y, seg_right - seg_left, h]
            self.new_boxes[l].append(seg_box)
            if use_cnn:
                vect = _normalize_to_32x32(sub_letter)
            else:
                vect = normalize_and_extract_features(sub_letter, debug_char_idx=i, debug_coordinates=seg_box, debug_line=l)
            self.vectors[l].append(vect)
            _dbg(f"[PROJECTION SPLIT] Added sub-image {seg_idx}: box={seg_box}, shape={sub_letter.shape}")
        return True

    def _auto_segment_box(self, l, i, letter, orig_box, oo_scale_l, chars):
        """Fallback automatic segmentation via _sample_widths_method over a range of
        candidate char counts; appends the best recognition results."""
        all_choices = []
        for c in range(int(chars), 0, -1):
            line_num = l if self.line_info.shapes.detect_o else None
            all_choices.append(self._sample_widths_method(c, letter, orig_box, oo_scale_l, line_num=line_num))
        mx = max(all_choices)
        for v in mx[-1]:
            self.new_boxes[l].append(v)
            self.vectors[l].append(v)
            self.line_info.shapes.img_arr[v[1]:v[1]+v[3], v[0]+v[2]] = 1

    def _normal_vect(self, l, i, letter, lindices, len_lindices, box, cache_miss_exc):
        """Shared body of the normal single-char vector paths. `cache_miss_exc` is
        the exception swallowed on the single-index feature-cache read — the only
        deliberate per-path difference: broad for the large-box fall-through,
        narrow (KeyError) for the small-box path."""
        x, y, w, h = box
        self.new_boxes[l].append([x, y, w, h])
        if use_cnn:
            vect = _normalize_to_32x32(letter)
        elif len_lindices == 1:
            try:
                vect = self.cached_features[lindices[0]]
            except cache_miss_exc:
                vect = normalize_and_extract_features(letter, debug_char_idx=i, debug_coordinates=[x, y, w, h], debug_line=l)
        else:
            vect = normalize_and_extract_features(letter, debug_char_idx=i, debug_coordinates=[x, y, w, h], debug_line=l)
        self.vectors[l].append(vect)

    def _normal_vect_large(self, l, i, letter, lindices, len_lindices, box):
        """Normal single-char vector for the large-box fall-through (broad cache-miss catch)."""
        self._normal_vect(l, i, letter, lindices, len_lindices, box, Exception)

    def _normal_vect_small(self, l, i, letter, lindices, len_lindices, box):
        """Normal single-char vector for the small-box path (narrow KeyError catch)."""
        self._normal_vect(l, i, letter, lindices, len_lindices, box, KeyError)

    def _process_stochastic_box(self, l, i, final, scale_l, oo_scale_l, breakwidth):
        ## New draw, takes into account tree hierarchy of contours
        final_boxes, final_indices, final_box_info = final
        x, y, w, h = final_boxes[l][i]
        lindices = final_indices[l][i]
        len_lindices = len(lindices)

        letter = self._draw_char_letter((x, y, w, h), lindices)

        if w*scale_l >= 1 and h*scale_l >= 1:
            letter = cv.resize(letter, dsize=(0,0), fx=scale_l, fy=scale_l)

        conservative_breakpoint = self._conservative_breakpoint(final_box_info, breakwidth, letter)

        if letter.shape[1] < conservative_breakpoint:   # normal-sized box
            self._normal_vect_small(l, i, letter, lindices, len_lindices, (x, y, w, h))
            return

        # Box too large — try to break it.
        sw = w*scale_l
        sh = h*scale_l
        chars = sw // (final_box_info.char_mean - 1.5*final_box_info.char_std)  # floor division
        chars = min(chars, 4)
        _dbg(f"[SEGMENT DEBUG] Large character detected: sw={sw:.1f}, calculated_chars={chars}")

        # Only segment if we're confident it's clearly more than one character.
        if chars > 1.5:
            # Vertical projection split first (works in the daemon, no GUI needed).
            # Uses the ORIGINAL box (w,h before the sw/sh reassignment below).
            split_points = find_projection_split_points(letter, final_box_info.char_mean)
            if split_points and self._projection_split_box(l, i, letter, (x, y, w, h), split_points):
                return   # sub-images already appended; skip normal processing
            _dbg(f"[PROJECTION SPLIT] No clear split points found for {letter.shape[1]}px blob, using automatic segmentation")
            # Fallback: automatic segmentation.
            self._auto_segment_box(l, i, letter, final_boxes[l][i], oo_scale_l, chars)
            w = sw
            h = sh

        # Normal processing (also the fall-through when chars <= 1.5).
        self._normal_vect_large(l, i, letter, lindices, len_lindices, (x, y, w, h))

    def _naro_char_with_o(self, i, k):
        """Predict the char for box (i,k) and append the naro vowel 'ོ'."""
        probs = predict_log_proba(self.vectors[i][k])
        mx = np.argmax(probs)
        prob = probs[0][mx]
        if use_cnn:
            ch = label_chars[int(predictor.classes_[mx])] + 'ོ'
        else:
            ch = label_chars[mx] + 'ོ'
        self.vectors[i][k] = ch
        return prob, ch

    def _merge_naro_into_box(self, i, k, box, box1):
        """Combine naro `box` with char `box1` at (i,k), stamping the 'ོ' vowel
        onto whichever vector representation this box currently holds."""
        try:
            nbox = list(combine_many_boxes([box, box1]))
        except:
            print((nbox, 'slkfjlkfj'))
            raise
        if isinstance(self.vectors[i][k], str):
            self.vectors[i][k] += 'ོ'
            nbox = box1
            nbox[-1] = self.vectors[i][k]
        elif isinstance(self.vectors[i][k], list):
            if not self.vectors[i][k][-1][-1] == 'ོ':
                pchar = self.vectors[i][k][-1] + 'ོ'
                self.vectors[i][k][-1] = pchar
            nbox = self.vectors[i][k]
        else:
            prob, ch = self._naro_char_with_o(i, k)
            nbox.append(prob)
            nbox.append(ch)
        self.new_boxes[i][k] = nbox

    def _combine_naros_stochastic(self):
        """detect_o post-pass: fold each detected naro (ོ) contour into the char
        box it overlaps. (Only runs when conf['detect_o'] is on.)"""
        for i, line in enumerate(self.new_boxes):
            used_boxes = set()
            for n in self.line_info.line_naros[i]:
                if n in used_boxes:
                    continue
                box = self.line_info.get_box(n)
                x,y,w,h = box
                for k, box1 in enumerate(line):
                    assert isinstance(box1, (list, tuple)), 'error - {}-{}-{}'.format(str(box1), i, k)
                    assert isinstance(box, (list, tuple)), box
                    try:
                        overlap = check_for_overlap(box1, box)
                    except:

                        print((i, k, box1, 'BOX problem'))
                    if overlap:
                        used_boxes.add(n)
                        self._merge_naro_into_box(i, k, box, box1)




