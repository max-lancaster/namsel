# encoding: utf-8
"""ExperimentalSegmenterMixin: the experimental-segmenter methods of Segmenter,
split out of segment.py to keep that file within the file-size limit.

Mixed into Segmenter via MRO, so the methods reach the host's own attributes and
methods through self; segment-module globals (classifier hooks, HMM matrices, the
ExpSegContext bundle, module helpers) are reached through `_s.*`. The dual
relative/absolute import mirrors segment's own style so both the packaged and the
flat (cwd=namsel_ocr) paths resolve to the same segment module object.
"""
import cv2 as cv
import numpy as np
from numpy import argmax, floor

try:
    from .fast_utils import ftrim, fadd_padding
    from .config_manager import default_config
    from . import segment as _s
except ImportError:
    from fast_utils import ftrim, fadd_padding
    from config_manager import default_config
    import segment as _s


class ExperimentalSegmenterMixin(object):
    """Experimental-segmenter methods for Segmenter (see module docstring)."""

    @staticmethod
    def _exp_split_segs(letter, chars, z, char_mean_int, vsum):
        """Slice `letter` into `chars` segments at ink-valley breaklines for one
        (chars, z) candidate. Returns the list of segment sub-images."""
        segs = []
        prev_breakline = 0
        for pos in range(int(chars-1)):
            if char_mean_int - z >= 0:
                upper_range = [int(np.round((pos+1)*(char_mean_int-z))), int(np.round((pos+1)*(char_mean_int+z)))]
                vsum_range = vsum[upper_range[0]:upper_range[1]]
                if vsum_range.any():
                    breakline = int(np.round((pos+1)*(char_mean_int-z) + argmax(vsum_range)))
                else:
                    breakline = None
                if breakline:
                    sg = letter[:, prev_breakline:breakline]
                    prev_breakline = breakline
                else:
                    sg = letter[:, int(np.round(pos*(char_mean_int-z))):int(np.round((pos+1)*(char_mean_int-z)))]
                    prev_breakline = int(np.round((pos+1)*(char_mean_int-z)))
                segs.append(sg)
        segs.append(letter[:, int(np.round((chars-1)*(char_mean_int-z))):])
        return segs

    def _exp_trim_segs(self, segs, seg_bxs, x, y, scale_l, final_box_info):
        """Fill sub-tsek noise, trim, and compute the scaled box for each segment.
        Returns (bxs, nsegs)."""
        bxs = []
        nsegs = []
        prev_w = 0
        for zi, ltb in enumerate(seg_bxs):
            seg = segs[zi]
            for b in ltb:
                if b[2] < (final_box_info.tsek_mean + 4*final_box_info.tsek_std) or b[3] < final_box_info.tsek_mean + 4*final_box_info.tsek_std:
                    seg[b[1]-1:b[1]+b[3]+1, b[0]-1:b[0]+b[2]+1] = True
            seg, ofst = ftrim(seg, new_offset=True)
            bx = [x+prev_w+(ofst['left']/scale_l), y + (ofst['top']/scale_l), seg.shape[1]/scale_l, seg.shape[0]/scale_l]
            prev_w += seg.shape[1]/scale_l
            bxs.append(bx)
            nsegs.append(seg)
        return bxs, nsegs

    @staticmethod
    def _exp_seg_boxes(segs):
        # OpenCV version-agnostic (3.x 3-tuple / 4.x 2-tuple)
        seg_ctrs = [cv.findContours(sg.copy(), mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE)[-2:] for sg in segs]
        # Explicit loop so the failing `sgc` is in scope for the diagnostic (in a
        # comprehension it wasn't, so the original except raised NameError instead).
        seg_bxs = []
        for sgc in seg_ctrs:
            try:
                seg_bxs.append([cv.boundingRect(k) for k in sgc[0]])
            except Exception:
                print(sgc)
                raise
        return seg_bxs

    def _exp_extract_xt(self, nsegs, bxs, l):
        if _s.use_cnn:
            return [_s._normalize_to_32x32(sg) for zi, sg in enumerate(nsegs) if 0 not in sg.shape]
        return [_s.normalize_and_extract_features(sg, debug_char_idx=zi, debug_coordinates=bxs[zi], debug_line=l)
                for zi, sg in enumerate(nsegs) if 0 not in sg.shape]

    def _experimental_seg_pass(self, l, letter, chars, z, ctx):
        """One (chars, z) segmentation candidate. Returns (prob, prds, bxs, xt)."""
        char_mean_int, vsum, x, y, scale_l, final_box_info = ctx
        segs = self._exp_split_segs(letter, chars, z, char_mean_int, vsum)
        segs = [fadd_padding(sg, 2) for sg in segs]
        seg_bxs = self._exp_seg_boxes(segs)
        bxs, nsegs = self._exp_trim_segs(segs, seg_bxs, x, y, scale_l, final_box_info)
        xt = self._exp_extract_xt(nsegs, bxs, l)
        prd_probs = _s.predict_log_proba(xt).astype(np.float32)
        prob, prds = _s.viterbi_cython(prd_probs.shape[0], _s.n_states, _s.start_p, _s.trans_p, prd_probs)
        return np.exp(prob), prds, bxs, xt

    def _break_experimental(self, l, letter, chars, ctx):
        """Search (chars, z) candidates for the best-Viterbi break of a wide box.
        Returns (best_seq, best_box_dim, best_prob)."""
        best_box_dim = []
        best_prob = 0.0
        best_seq = None
        for c in range(int(chars), 1, -1):
            for z in range(0, 21, 2):
                prob, prds, bxs, xt = self._experimental_seg_pass(l, letter, c, z, ctx)
                if prob > best_prob:
                    best_prob = prob
                    best_seq = prds
                    best_box_dim = bxs
        if not best_box_dim:
            # Fallback: last candidate's result (matches the original's use of the
            # loop-final prob/prds/bxs when nothing beat best_prob).
            best_prob = prob
            best_seq = prds
            best_box_dim = bxs
        return best_seq, best_box_dim, best_prob

    def _append_experimental_result(self, l, best_seq, best_box_dim, best_prob):
        for u in range(len(best_seq)):
            self.vectors[l].append(_s.label_chars[best_seq[u]])
            best_box = best_box_dim[u]
            best_box = [int(np.round(ii)) for ii in best_box]
            best_box.append(best_prob)
            best_box.append(_s.label_chars[best_seq[u]])
            self.new_boxes[l].append(best_box)
            try:
                self.line_info.shapes.img_arr[best_box[1]:best_box[1]+best_box[3], best_box[0]+best_box[2]] = 1
            except Exception as e:
                # separator column lands outside the page crop — harmless, skip the marker
                _dbg(f"[EXP-SEG] separator draw out of bounds: {e}")

    def _experimental_normal_vect(self, l, i, letter, box):
        x, y, w, h = box
        self.new_boxes[l].append([x, y, w, h])
        if _s.use_cnn:
            vect = _s._normalize_to_32x32(letter)
        else:
            vect = _s.normalize_and_extract_features(letter, debug_char_idx=i, debug_coordinates=[x, y, w, h], debug_line=l)
        self.vectors[l].append(vect)

    def _process_experimental_box(self, l, i, final_boxes, final_indices, scale_l, char_mean_int, final_box_info):
        x, y, w, h = final_boxes[l][i]
        letter = self._draw_char_letter((x, y, w, h), final_indices[l][i])
        letter = cv.resize(letter, dsize=(0,0), fx=scale_l, fy=scale_l)

        breakpoint = final_box_info.char_mean + self.breakwidth*final_box_info.char_std
        if letter.shape[1] < breakpoint:   # not too large
            self._experimental_normal_vect(l, i, letter, (x, y, w, h))
            return

        sw = w*scale_l
        sh = h*scale_l
        vsum = letter.sum(axis=0)
        chars = sw // (final_box_info.char_mean - 1.5*final_box_info.char_std)  # floor division
        if not (10.0 > chars > 1.0):   # don't break spans > 10 chars
            self._experimental_normal_vect(l, i, letter, (x, y, w, h))
            return

        ctx = _s.ExpSegContext(char_mean_int, vsum, x, y, scale_l, final_box_info)
        best_seq, best_box_dim, best_prob = self._break_experimental(l, letter, chars, ctx)
        self._append_experimental_result(l, best_seq, best_box_dim, best_prob)

    def _process_experimental_line(self, l, final_boxes, final_indices, final_box_info, scales):
        try:
            scale_l = scales[l]
        except:
            print(('ERROR AT ', l, len(scales)))
            raise
        char_mean_int = floor(final_box_info.char_mean)

        try:
            lb = list(range(len(final_indices[l])))
        except IndexError:
            print('index error')
            return

        for i in lb: # for each line box
            self._process_experimental_box(l, i, final_boxes, final_indices, scale_l, char_mean_int, final_box_info)

    def construct_vector_set_experimental(self):
        _s.build_vector_set(self, self._process_experimental_line, self._combine_naros_experimental)

    def _experimental_naro_char(self, i, k, prob):
        """Stamp the naro vowel onto vector (i,k) — experimental path (has an
        rbfcls fallback when the CNN isn't in use)."""
        probs = _s.predict_log_proba(self.vectors[i][k])
        mx = np.argmax(probs)
        prob = probs[0][mx]
        if _s.use_cnn:
            ch = _s.label_chars[int(_s.predictor.classes_[mx])] + 'ོ'
        else:
            mx = self.line_info.rbfcls.predict(self.vectors[i][k])[0]
            ch = _s.label_chars[mx] + 'ོ'
        self.vectors[i][k] = ch
        return prob, ch

    def _combine_naros_experimental(self):
        for i, l in enumerate(self.new_boxes):
            for n in self.line_info.line_naros[i]:
                box = self.line_info.get_box(n)
                x, y, w, h = box
                r0 = x+w
                for k, b in enumerate(l):
                    # Gap-tolerant overlap for compound syllables (small gaps between
                    # naro and main character allowed, like Brazilian diacritics).
                    gap_size = max(0, max(b[0] - r0, x - (b[0]+b[2])))
                    char_mean = getattr(self.line_info.shapes, 'char_mean', 15.0)
                    gap_tolerance = char_mean * default_config.get('compound_syllable_gap_tolerance', 0.15)
                    if gap_size <= gap_tolerance:
                        overlap_ratio = 1.0
                    else:
                        overlap_ratio = ((b[2] + w) - abs(b[0] - x) - abs((b[0]+b[2]) - r0)) / (2*float(min(w, b[2])))
                    if overlap_ratio <= .8:
                        continue
                    try:
                        nbox = list(_s.combine_many_boxes([box, b]))
                    except:
                        print((nbox[3]))
                        raise
                    if isinstance(self.vectors[i][k], str):
                        self.vectors[i][k] += 'ོ'
                        nbox = b
                        nbox[-1] = self.vectors[i][k]
                    else:
                        prob, ch = self._experimental_naro_char(i, k, None)
                        nbox.append(prob)
                        nbox.append(ch)
                    self.new_boxes[i][k] = nbox
