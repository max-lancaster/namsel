#! /usr/bin/python
# encoding: utf-8
"""Small-char placement + box-finalization helpers for recognize_chars_probout.

Split out of recognize.py to keep that file within the file-size limit. These
helpers read a few recognize-module globals (character map, classifier hooks)
via `_r.*`; the import mirrors recognize's own relative/absolute dual style so
both the packaged and the flat (cwd=namsel_ocr) import paths resolve to the same
recognize module object.
"""
from collections import namedtuple

from bisect import bisect
from cv2 import drawContours
import numpy as np

try:
    from . import recognize as _r
except ImportError:
    import recognize as _r

# Vertical extent + left/right/mid x of the local box neighborhood around an
# insertion point, bundled so the tsek-placement helpers stay within the
# parameter limit.
BaselineBounds = namedtuple("BaselineBounds", "top bottom left right mid local_span")

# One line's mutable recognition stream, shared by the small-char placement
# helpers (kept in sync by every insertion): the vector list, the box list, the
# left-edge index, and the accumulated tsek widths.
LineStream = namedtuple("LineStream", "vectors new_boxes left_edges tsek_widths")


def _recompute_feature_vect(segmentation, s, x, y, w, h):
    """Cache-miss path: rasterize contour s into an h x w char image and return
    its feature vector (CNN 32x32 or sklearn hand-features)."""
    cnt = segmentation.line_info.shapes.contours[s]
    char_arr = np.ones((h, w), dtype=np.uint8)
    drawContours(char_arr, [cnt], -1, 0, thickness=-1, offset=(-x, -y))
    if _r.use_cnn:
        from namsel_ocr.segment import _normalize_to_32x32
        return _normalize_to_32x32(char_arr)
    return _r.normalize_and_extract_features(char_arr)


def _classify_small_char(segmentation, s, x, y, w, h, cached_features, cached_pred_prob):
    """Classify a small connected component: punctuation (tsheg ་ / shad ། recovered
    from small_contour_indices) vs a regular character. Returns (prd, prob)."""
    if s in segmentation.line_info.shapes.small_contour_indices:
        # Detected as punctuation — recover its original tsheg/shad prediction;
        # default to tsheg when unclear or unavailable (backward compat).
        try:
            inx, _ = cached_pred_prob[s]
            original_pred = _r.dig_to_char[inx] if inx in _r.dig_to_char else "?"
            if original_pred == "།":
                return "།", 1.0
            return "་", 1.0
        except:
            return "་", 1.0
    # Regular character — classify normally.
    try:
        feature_vect = cached_features[s]   # guard: KeyError here → recompute below
        inx, probs = cached_pred_prob[s]
        return _r.dig_to_char[inx], probs[inx]
    except:
        return _r.prd_prob(_recompute_feature_vect(segmentation, s, x, y, w, h))


def _insert_char_at(vectors, new_boxes, left_edges, pos, prd, prob, bx):
    """Insert a classified char (prd/prob/bx) at pos, keeping vectors / new_boxes /
    left_edges in sync. Box becomes [x, y, w, h, prob, char]."""
    vectors.insert(pos, prd)
    new_boxes.insert(pos, bx)
    new_boxes[pos].append(float(prob))  # Python float, not numpy
    new_boxes[pos].append(str(prd))     # character as string
    left_edges.insert(pos, bx[0])


def _baseline_bounds(new_boxes, insertion_pos, left_items, right_items):
    """Vertical extent + left/right/mid x of the local box neighborhood around the
    insertion point. Returns (top, bottom, left, right, mid, local_span)."""
    top = 1000000  # arbitrary high number
    bottom = 0
    lower = max(insertion_pos - left_items, 0)
    upper = min(len(new_boxes)-1, insertion_pos+right_items)
    left = new_boxes[lower][0]
    right = new_boxes[upper][0] + new_boxes[upper][2]
    mid = new_boxes[insertion_pos][0] + new_boxes[insertion_pos][2] if insertion_pos < len(new_boxes) else right
    for j in new_boxes[lower:upper]:
        if j[1] < top:
            top = j[1]
        if j[1] + j[3] > bottom:
            bottom = j[1] + j[3]
    local_span = bottom - top
    top, bottom, left, right, mid = [int(np.round(ff)) for ff in [top, bottom, left, right, mid]]
    return BaselineBounds(top, bottom, left, right, mid, local_span)


def _local_baselines(img_arr, top, bottom, left, right, mid, local_span):
    """Local baseline y (min-ink row) on the left and right of the insertion point;
    falls back to the band midpoint if the ink projection can't be taken."""
    try:
        left_sum = img_arr[top:bottom, left:mid].sum(axis=1)
        right_sum = img_arr[top:bottom, mid:right].sum(axis=1)
        bl_left = top + left_sum.argmin()
        bl_right = top + right_sum.argmin() if mid != right else bl_left
    except (IndexError, ValueError):
        bl_left = top + local_span // 2
        bl_right = bl_left
    return bl_left, bl_right


def _tsek_is_valid(img_arr, bx, insertion_pos, vectors, bounds):
    """Whether a tsek at bx sits plausibly on the local baseline / middle band (a
    permissive OR of baseline-hit, middle-band, end-of-line, or small-tsek)."""
    top, bottom, left, right, mid, local_span = bounds
    bl_left, bl_right = _local_baselines(img_arr, top, bottom, left, right, mid, local_span)
    char_middle_y = bx[1] + bx[3] // 2
    line_middle_y = top + local_span // 2
    baseline_tolerance = local_span * 0.4  # 40% tolerance around baseline/middle
    baseline_hit = ((bl_left >= bx[1] and bl_left <= bx[1] + bx[3]) or
                    (bl_right >= bx[1] and bl_right <= bx[1] + bx[3]))
    return (baseline_hit or
            abs(char_middle_y - line_middle_y) <= baseline_tolerance or
            insertion_pos == len(vectors) or
            bx[3] <= local_span * 0.3)


def _adjust_tsek_pos(insertion_pos, new_boxes, x, w):
    """A tsek can render just before its indicated position; nudge left one slot
    when it overlaps a much-wider previous box."""
    if insertion_pos <= len(new_boxes):
        prev_box = new_boxes[insertion_pos-1]
        if 0 <= x - prev_box[0] < w and 2*w < prev_box[2]:
            return insertion_pos - 1
    return insertion_pos


def _place_small_char_baseline(img_arr, bx, prd, prob, insertion_pos, left_items, right_items, stream):
    """Baseline-method placement of one small char: put a valid tsek on the
    baseline, append an invalid one onto the existing box, or insert other
    punctuation when it fits the local vertical band."""
    vectors, new_boxes, left_edges, tsek_widths = stream
    x, y, w, h = bx[:4]
    bounds = _baseline_bounds(new_boxes, insertion_pos, left_items, right_items)
    top, bottom, left, right, mid, local_span = bounds
    if prd == '་' and local_span > 0:
        if _tsek_is_valid(img_arr, bx, insertion_pos, vectors, bounds):
            insertion_pos = _adjust_tsek_pos(insertion_pos, new_boxes, x, w)
            _insert_char_at(vectors, new_boxes, left_edges, insertion_pos, prd, prob, bx)
        else:
            new_boxes[insertion_pos].append(float(prob))
            new_boxes[insertion_pos].append(str(prd))
            left_edges.insert(insertion_pos, bx[0])
            tsek_widths.append(bx[2])
    elif (bx[1] >= top - .25*local_span and bx[1] + bx[3] <= bottom + local_span*.25) or (insertion_pos == len(vectors)):
        _insert_char_at(vectors, new_boxes, left_edges, insertion_pos, prd, prob, bx)


def _place_small_char(segmentation, s, stream, img_arr, tsek_insert_method,
                      cached_features, cached_pred_prob):
    """Classify and insert one small connected-component (tsek/shad/punctuation)
    at its left-edge-sorted position on the line."""
    vectors, new_boxes, left_edges, tsek_widths = stream
    bx = list(segmentation.line_info.shapes.get_boxes()[s])
    x, y, w, h = bx
    prd, prob = _classify_small_char(segmentation, s, x, y, w, h, cached_features, cached_pred_prob)
    insertion_pos = bisect(left_edges, x)
    # More left neighbors near line end, more right neighbors near line start, to
    # get enough boxes to define the local baseline.
    left_items, right_items = (12, 5) if insertion_pos >= len(new_boxes) else (6, 12)

    if tsek_insert_method != 'baseline':
        _insert_char_at(vectors, new_boxes, left_edges, insertion_pos, prd, prob, bx)
        return
    _place_small_char_baseline(img_arr, bx, prd, prob, insertion_pos, left_items, right_items, stream)


def _should_insert_space(new_boxes, i, tsek_mean):
    """Scale-adaptive spacing detection between boxes i and i+1 (OR logic — any
    criterion triggers a space): a traditional large gap, a large gap relative to
    char width, or a substantial missing-tsek gap."""
    gap_width = new_boxes[i+1][0] - (new_boxes[i][0] + new_boxes[i][2])
    avg_char_width = (new_boxes[i][2] + new_boxes[i+1][2]) / 2
    scale_factor = max(1.0, avg_char_width / 15.0)  # 15px = standard char width
    min_gap_threshold = max(4, int(6 * scale_factor))
    min_large_gap_threshold = max(6, int(8 * scale_factor))
    if gap_width >= 2*tsek_mean:
        return True
    if gap_width >= 1.5 * avg_char_width and gap_width >= min_large_gap_threshold:
        return True
    if gap_width >= 1.2 * tsek_mean and gap_width >= min_gap_threshold and gap_width >= 0.6 * avg_char_width:
        return True
    return False


def _normalize_processed_box(new_boxes, i):
    """Trim an already-completed box (>6 elements from the tsheg pass) back to a
    clean [x, y, w, h, prob, char]. Leaves an exactly-6 box unchanged."""
    box = new_boxes[i]
    if len(box) == 7:
        new_boxes[i] = box[:4] + [float(box[4]), str(box[6])]
    elif len(box) > 7:
        prob = box[4] if len(box) > 4 else 1.0
        new_boxes[i] = box[:4] + [float(prob), str(box[-1])]


def _classify_box(new_boxes, i, v):
    """Classify an unprocessed box's vector (or use its pre-segmented string) and
    append (prob, char)."""
    if not isinstance(v, str):
        try:
            prd, prob = _r.prd_prob(v)
        except:
            print(f"Error processing vector {i}: {v}")
            prd, prob = '?', 0.5
    else:
        prd = v
        prob = .95
    if len(new_boxes[i]) < 6:
        try:
            new_boxes[i].append(float(prob))
        except:
            new_boxes[i].append(1.0)
        new_boxes[i].append(str(prd))


def _finalize_box(new_boxes, i, v):
    """Ensure new_boxes[i] is a well-formed [x, y, w, h, prob, char] entry. Shared
    by the spacing loop and the last-vector tail (previously duplicated verbatim)."""
    if len(new_boxes[i]) >= 6:
        _normalize_processed_box(new_boxes, i)
    else:
        _classify_box(new_boxes, i, v)

    if len(new_boxes[i]) != 6:
        print(f"[ERROR] Box {i} has wrong format: {new_boxes[i]}")
    if not isinstance(new_boxes[i][-1], str):
        print(f"[ERROR] Box {i} character is not string: {type(new_boxes[i][-1])} = {new_boxes[i][-1]}")


def _insert_emph_markers(segmentation, vectors, new_boxes, left_edges, emph_markers):
    """Insert emphasis markers at their left-edge-sorted positions on the line."""
    for em in emph_markers:
        bx = list(segmentation.line_info.shapes.get_boxes()[em])
        mkinx = segmentation.line_info.shapes.cached_pred_prob[em][0]
        marker = _r.dig_to_char[mkinx]
        marker_prob = segmentation.line_info.shapes.cached_pred_prob[em][1][mkinx]
        bx.append(marker_prob)
        bx.append(marker)
        insertion_pos = bisect(left_edges, bx[0])
        vectors.insert(insertion_pos, marker)
        new_boxes.insert(insertion_pos, bx)
        left_edges.insert(insertion_pos, bx[0])


def _emit_line_chars(vectors, new_boxes, tsek_mean):
    """Walk a line's vectors, emitting each finalized [x,y,w,h,prob,char] box and a
    space where the gap warrants one. Returns the line's tmp_result."""
    tmp_result = []
    for i, v in enumerate(vectors[:-1]):
        insert_space = _should_insert_space(new_boxes, i, tsek_mean)
        _finalize_box(new_boxes, i, v)
        tmp_result.append(new_boxes[i])
        if insert_space:
            tmp_result.append([-1, -1, -1, -1, 1.0, ' '])  # space char as string
    # Last vector (same unified pathway, no trailing space).
    if len(vectors) > 0:
        i = len(vectors) - 1
        _finalize_box(new_boxes, i, vectors[i])
        tmp_result.append(new_boxes[i])
    return tmp_result


def _process_probout_line(segmentation, l, vectors, tsek_mean,
                          cached_features, cached_pred_prob, tsek_insert_method):
    """Build one line's recognized-char result: insert small chars (tsek/shad) and
    emphasis markers into the box stream, then emit chars + spaces."""
    new_boxes = segmentation.new_boxes[l]
    small_chars = segmentation.line_info.small_cc_lines_chars[l]
    # Line Cut has no emph_lines object, so work around it for now.
    emph_markers = getattr(segmentation.line_info, 'emph_lines', [])
    if emph_markers:
        emph_markers = emph_markers[l]
    img_arr = segmentation.line_info.shapes.img_arr
    left_edges = [b[0] for b in new_boxes]
    tsek_widths = []

    # Consider small chars from the end of the line backward (useful for misplaced
    # tseks, and maybe for TOC — though that should be checked).
    stream = LineStream(vectors, new_boxes, left_edges, tsek_widths)
    for s in small_chars[::-1]:
        _place_small_char(segmentation, s, stream, img_arr, tsek_insert_method,
                          cached_features, cached_pred_prob)

    _insert_emph_markers(segmentation, vectors, new_boxes, left_edges, emph_markers)
    return _emit_line_chars(vectors, new_boxes, tsek_mean)


# ---------------------------------------------------------------------------
# HMM recognizer line helpers (recognize_chars_hmm path), split out of
# recognize.py for the same file-size reason.
# ---------------------------------------------------------------------------


def _label_to_char(inx):
    """Map a cached label index to its char, falling back to the nearest defined
    key at or below inx (or the replacement char) when unmapped."""
    if inx in _r.dig_to_char:
        return _r.dig_to_char[inx]
    valid_keys = [k for k in _r.dig_to_char.keys() if k <= inx]
    return _r.dig_to_char[max(valid_keys)] if valid_keys else '�'


def _hmm_small_char_info(segmentation, s, cached_features, cached_pred_prob):
    """Classify one small connected component into a position/char dict for the
    HMM combine pass."""
    bx = list(segmentation.line_info.shapes.get_boxes()[s])
    x, y, w, h = bx
    try:
        feature_vect = cached_features[s]   # guard: KeyError → recompute below
        inx, probs = cached_pred_prob[s]
        prob = probs[inx]
        prd = _label_to_char(inx)
    except:
        prd, prob = _r.prd_prob(_recompute_feature_vect(segmentation, s, x, y, w, h))
    return {'x': x, 'y': y, 'w': w, 'h': h, 'box': bx, 'pred': prd, 'prob': prob, 'original_index': s}


def _hmm_validate_small(info, scale_factor):
    """Scale-adaptive validity of a small char: tsheg must pass a size/position
    check; shads and other small chars are always valid."""
    if info['pred'] == '་':  # tsheg
        w, h, y = info['w'], info['h'], info['y']
        min_w = max(2, int(2 * scale_factor)); max_w = min(10, int(8 * scale_factor))
        min_h = max(2, int(2 * scale_factor)); max_h = min(20, int(15 * scale_factor))
        min_y = int(14 * scale_factor); max_y = int(220 * scale_factor)
        return (w >= min_w and w <= max_w and h >= min_h and h <= max_h and y >= min_y and y <= max_y)
    return True


def _hmm_build_positions(segmentation, vectors, new_boxes, small_chars, cached_features, cached_pred_prob):
    """Combine main-char vectors and validated small chars into one x-sorted list."""
    scale_factor = getattr(segmentation.line_info.shapes, 'global_scale_factor', 1.0)
    positions = []
    for i, vector in enumerate(vectors):
        if i < len(new_boxes):
            positions.append({'x': new_boxes[i][0], 'type': 'main', 'vector': vector, 'box': new_boxes[i], 'index': i})
    for s in small_chars:
        info = _hmm_small_char_info(segmentation, s, cached_features, cached_pred_prob)
        if _hmm_validate_small(info, scale_factor):
            positions.append({'x': info['x'], 'type': 'small', 'pred': info['pred'], 'box': info['box'],
                              'prob': info['prob'], 'original_index': info['original_index'], 'validated': True})
    positions.sort(key=lambda c: c['x'])   # stable → main precedes small at equal x
    return positions


def _hmm_rebuild(positions):
    """Rebuild (vectors, new_boxes, left_edges) from the x-sorted position list."""
    vectors, new_boxes, left_edges = [], [], []
    for char in positions:
        if char['type'] == 'main':
            vectors.append(char['vector'])
            new_boxes.append(char['box'])
            left_edges.append(char['x'])
        elif char['type'] == 'small' and char.get('validated', False):
            box_with_meta = char['box'] + [char['prob'], char['pred']]
            vectors.append(box_with_meta)
            new_boxes.append(box_with_meta)
            left_edges.append(char['x'])
    return vectors, new_boxes, left_edges


def _hmm_group_vectors(vectors, new_boxes, tmp_result):
    """Split vectors into contiguous runs of main-char vectors, breaking at
    pre-segmented strings/lists. Manual string labels emit directly to tmp_result.
    Returns (allstrs, allinx)."""
    allstrs, curstr, allinx, curinx = [], [], [], []
    for j, v in enumerate(vectors):
        if isinstance(v, (str, list)):
            allstrs.append(curstr)
            allinx.append(curinx)
            if isinstance(v, str):
                box = new_boxes[j] if j < len(new_boxes) else [0, 0, 0, 0]
                tmp_result.append(box + [1.0, v])  # high confidence for manual labels
            curstr, curinx = [], []
        else:
            curstr.append(v)
            curinx.append(j)
    allstrs.append(curstr)
    allinx.append(curinx)
    return allstrs, allinx


def _hmm_decode_group(group, n_states):
    """Viterbi-decode one contiguous run of main-char vectors. Returns (prds, probs)."""
    probs = _r.predict_log_proba(group)
    if len(probs) == 1:
        _classes = _r.predictor.classes_ if _r.use_cnn else _r.cls.classes_
        return [int(_classes[probs[0].argmax()])], probs
    probs = probs.astype(np.float32)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*np.int.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*numpy.*has no attribute.*int.*")
            _prb, prds = _r.viterbi_cython(len(probs), n_states, _r.start_p, _r.trans_p, probs)
    except (AttributeError, TypeError, DeprecationWarning):
        # NumPy-compat fallback: per-column argmax
        _classes = _r.predictor.classes_ if _r.use_cnn else _r.cls.classes_
        prds = [int(_classes[np.argmax(row)]) for row in probs]
    return prds, probs


def _hmm_write_group(new_boxes, prds, probs, inx_group):
    """Append (prob, char) to each box in a decoded group."""
    for c in range(len(prds)):
        ind = inx_group[c]
        new_boxes[ind].append(np.exp(probs[c].max()))
        new_boxes[ind].append(_r.dig_to_char.get(int(prds[c]), '�'))


def _hmm_emit(new_boxes, tmp_result, tsek_mean):
    """Emit each box, inserting a space where the gap to the next box is large."""
    for ind in range(len(new_boxes)):
        tmp_result.append(new_boxes[ind])
        if ind + 1 < len(new_boxes) and new_boxes[ind+1][0] - (new_boxes[ind][0] + new_boxes[ind][2]) >= 1.5*tsek_mean:
            tmp_result.append([-1, -1, -1, -1, 1.0, ' '])


def _process_hmm_line(segmentation, l, vectors, n_states, tsek_mean, cached_features, cached_pred_prob):
    """Build one line's recognized-char result via the HMM recognizer: combine +
    rebuild the char stream, Viterbi-decode each main-char run, then emit."""
    new_boxes = segmentation.new_boxes[l]
    small_chars = segmentation.line_info.small_cc_lines_chars[l]
    positions = _hmm_build_positions(segmentation, vectors, new_boxes, small_chars,
                                     cached_features, cached_pred_prob)
    vectors, new_boxes, _left_edges = _hmm_rebuild(positions)

    tmp_result = []
    allstrs, allinx = _hmm_group_vectors(vectors, new_boxes, tmp_result)
    for f, group in enumerate(allstrs):
        if not group:
            continue
        prds, probs = _hmm_decode_group(group, n_states)
        _hmm_write_group(new_boxes, prds, probs, allinx[f])
    _hmm_emit(new_boxes, tmp_result, tsek_mean)
    return tmp_result
