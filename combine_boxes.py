# encoding: utf-8
"""CombineBoxesForPage + its tsek-width helpers, split out of segment.py to keep
that file within the file-size limit.

CombineBoxesForPage groups a page's raw connected-component boxes into character
boxes (tsheg-aware). It calls two segment-resident helpers (_dbg, combine_many_boxes)
via `_s.*`; the import mirrors segment's own relative/absolute dual style so both
the packaged and the flat (cwd=namsel_ocr) paths resolve to the same segment module.
"""
import numpy as np
from numpy import mean, std

try:
    from .transitions import horizontal_transitions
    from .config_manager import default_config
    from . import segment as _s
except ImportError:
    from transitions import horizontal_transitions
    from config_manager import default_config
    import segment as _s


def box_attrs(b):
    led = b[0] # left edge
    red = b[0] + b[2] # right edge
    top = b[1]
    bottom = b[1] + b[3]
    return led, red, top, bottom, b[2], b[3]

#@profile


def _tsek_bounds(shapes, scale_factor):
    """(min_width, max_width) tsek bounds. Large (scale > 2) images get tighter
    bounds to avoid false positives; normal/small images get permissive ones."""
    if scale_factor > 2.0:
        return max(2, shapes.tsek_mean - 2*shapes.tsek_std), shapes.tsek_mean + 1*shapes.tsek_std
    return max(1, shapes.tsek_mean - 4*shapes.tsek_std), shapes.tsek_mean + 2*shapes.tsek_std


def _dbg_tsek(cur_ind, width, min_width, max_width, scale_factor, width_check, baseline_check):
    if not default_config.get('debug_output', False):
        return
    b = f"width={width}, bounds=[{min_width:.1f}, {max_width:.1f}], scale={scale_factor:.3f}"
    if width_check and baseline_check:
        print(f"[IN_TSEK_POP] Character {cur_ind}: {b} - DETECTED AS TSHEG")
    elif width_check:
        print(f"[IN_TSEK_POP] Character {cur_ind}: {b} - FAILED baseline check")
    elif baseline_check:
        print(f"[IN_TSEK_POP] Character {cur_ind}: {b} - FAILED width check")
    elif cur_ind == 34:
        print(f"[IN_TSEK_POP] Character {cur_ind}: {b} - NOT DETECTED")


def in_tsek_pop(shapes, width, topn, top, bottomn, baseline, cur_ind):
    '''Determine whether a box is of approx tsek-width'''
    # Scale-adaptive tsheg detection criteria
    if not (hasattr(shapes, 'tsek_mean') and shapes.tsek_mean > 0):
        return False

    scale_factor = getattr(shapes, 'global_scale_factor', 1.0)
    min_width, max_width = _tsek_bounds(shapes, scale_factor)

    baseline_tolerance = 1.0 * shapes.tsek_std if shapes.tsek_std > 0 else 5
    baseline_check = (topn - baseline_tolerance <= baseline <= bottomn + baseline_tolerance)
    width_check = (min_width <= width <= max_width)

    _dbg_tsek(cur_ind, width, min_width, max_width, scale_factor, width_check, baseline_check)

    # Note: no adaptive-bounds fallback — it caused false positives for legitimate
    # characters; small-contour detection in page_elements2.py handles that case.
    return bool(width_check and baseline_check)


class CombineBoxesForPage(object):
    def __init__(self, line_info):
        self.widths = []
        self.final_boxes = []
        self.final_indices = []
        self.line_width_means = []
        self.larger_font_lines = []
        self.line_info = line_info
        self.transitions = []
        self.hangoff = line_info.shapes.conf['combine_hangoff']

        for i in range(self.line_info.k):
            initial_boxes = len(line_info.lines_chars[i]) if i < len(line_info.lines_chars) else 0
            initial_chars = line_info.lines_chars[i] if i < len(line_info.lines_chars) else []
            if default_config.get('debug_output', False):
                print(f"[DEBUG] Line {i}: Starting with {initial_boxes} boxes before combination: {initial_chars}")
            self.combine_for_line_ind(self.line_info, lineind=i)
            final_boxes = len(self.final_boxes[i]) if i < len(self.final_boxes) else 0
            if default_config.get('debug_output', False):
                print(f"[DEBUG] Line {i}: {final_boxes} boxes after combination")


        ###########IMPORTANT USE THIS NORMALLY


        self.char_mean = mean(self.widths)
        self.char_std = std(self.widths)
        self.tsek_mean = line_info.shapes.tsek_mean
        self.tsek_std = line_info.shapes.tsek_std
        

    ###########################################
    # (Dead helpers removed 2026-07-02: _low_ink_sort and li_combine_for_line_ind
    # had no callers anywhere in the repo — only combine_for_line_ind is used.)

    @staticmethod
    def _line_extent(line_info, line):
        top = 1000000 # arbitrary high number
        bottom = 0
        for k in line:
            j = line_info.get_box(k)
            if j[1] < top:
                top = j[1]
            if j[1] + j[3] > bottom:
                bottom = j[1] + j[3]
        return top, bottom

    @staticmethod
    def _sum_whitespace(line_info, line):
        whitespace = np.zeros(len(line), dtype=int)
        for p, c in enumerate(line):
            if p + 1 < len(line):
                ab = line_info.get_box(c)
                nab = line_info.get_box(line[p+1])
                ws_diff = nab[0] - (ab[0]+ab[2])
                whitespace[p] = ws_diff
        return whitespace.sum()

    @staticmethod
    def _blank_small_ccs(line_info, lineind, ln_arr, top, firstbox):
        #             Remove small contours when calculating scale
        try:
            for inx in line_info.small_cc_lines_chars[lineind]:
                box = line_info.get_box(inx)
                ln_arr[box[1] - top:box[1]+box[3]-top, box[0]-firstbox[0]:box[0]+box[2]-firstbox[0]] = 1
        except Exception as e:
            _s._dbg(f"[segment] small-cc blanking skipped for line {lineind}: {e}")

    def _line_scale(self, line_info, lineind, line):
        """Per-line scale factor from horizontal ink transitions (line non-empty)."""
        shapes = line_info.shapes
        top, bottom = self._line_extent(line_info, line)

        firstbox = line_info.get_box(line[0])
        lastbox = line_info.get_box(line[-1])
        sum_whitespace = self._sum_whitespace(line_info, line)

        ln_arr = shapes.img_arr[top:bottom,firstbox[0]:lastbox[0]+lastbox[2]].copy()

        self._blank_small_ccs(line_info, lineind, ln_arr, top, firstbox)

        # Convert to uint8 for horizontal_transitions compatibility
        ln_arr_uint8 = (ln_arr * 255).astype(np.uint8) if ln_arr.dtype != np.uint8 else ln_arr
        crss_denom = float(max(horizontal_transitions(ln_arr_uint8)))
        if crss_denom > 0:
            crossings_val = (ln_arr.shape[1]-sum_whitespace )/crss_denom
            scale =  35.0 / crossings_val
        else:
            scale = 1.0
        return scale

    @staticmethod
    def _log_combine_flags(line_info, i, next_attrs, raw_tsheg, small_tsheg, is_tsheg, tsheg_separation):
        ledn, ren, topn, bottomn, wn, hn = next_attrs
        if i == 34:
            _s._dbg(f"[SEGMENT DEBUG] Character 34: is_tsheg={raw_tsheg}, is_small_contour_tsheg={small_tsheg}")
        if is_tsheg:
            _s._dbg(f"[TSHEG DEBUG] Character {i} detected as tsheg: width={wn}, height={hn}, tsek_mean={line_info.shapes.tsek_mean}, height_threshold={1.5 * line_info.shapes.tsek_mean:.1f}")
        if tsheg_separation:
            _s._dbg(f"[TSHEG SEPARATION] Separating tsheg at index {i}, box=({ledn},{topn},{wn},{hn})")

    @staticmethod
    def _combine_flags(line_info, i, cur_attrs, next_attrs):
        led, red, top, bottom, w, h = cur_attrs
        ledn, ren, topn, bottomn, wn, hn = next_attrs

        # If left edge of next box doesn't overlap cur box
        # separate as 2 different chars
        if not isinstance(i, str):
            is_interior = line_info.shapes.hierarchy[0][i][0] < 0 and line_info.shapes.hierarchy[0][i][1] < 0 and line_info.shapes.hierarchy[0][i][2] < 0 # i.e. it has no peers at its place in the tree... and no children
        else:
            is_interior = False # Its a string, meaning it is the result of a horizontal cut and likely not an interior

        # Check if characters should be separated (gap between them OR current character is a tsheg)
        raw_tsheg = in_tsek_pop(line_info.shapes, wn, topn, top, bottomn, bottom, i)

        # Enhanced tsheg detection: also check if character is in small_contour_indices
        small_tsheg = hasattr(line_info.shapes, 'small_contour_indices') and i in line_info.shapes.small_contour_indices

        # Combine both detection methods
        is_tsheg = raw_tsheg or small_tsheg

        tsheg_separation = (not is_interior and is_tsheg and not hn > 1.5*line_info.shapes.tsek_mean)

        CombineBoxesForPage._log_combine_flags(line_info, i, next_attrs, raw_tsheg, small_tsheg, is_tsheg, tsheg_separation)

        return is_interior, is_tsheg, tsheg_separation

    def _should_separate(self, is_interior, is_tsheg, cur_attrs, next_attrs, i):
        """The original if/elif absorb-or-separate chain, evaluated only when the
        next box overlaps the current group (ledn <= red) and no tsheg_separation."""
        led, red, top, bottom, w, h = cur_attrs
        ledn, ren, topn, bottomn, wn, hn = next_attrs
        # TSHEG PROTECTION: Check if current character is a tsheg before combining
        # If it's a tsheg, treat it as separate even if spatially close
        if is_tsheg:
            _s._dbg(f"[TSHEG PROTECTION] Preventing combination of tsheg at index {i}, treating as separate")
            return True
        # one box is completely enveloped by the other
        if is_interior:
            return False
        if ((float(min(wn, w)) - abs((red - ledn))) / float(min(wn, w))) < self.hangoff: # amount hanging off end is 30 %, but protect tshegs
            _s._dbg(f"[TSHEG PROTECTION] Hangoff condition allows combining for character {i} (not a tsheg)")
            return False
        # The overlap is incidental / boxes are not related
        _s._dbg(f"[CHAR SEPARATION] Incidental overlap, separating character {i}")
        return True

    def _flush_group(self, line_info, fli, flb, cur_ind, scale):
        fli.append(cur_ind)
        x,y,w,h = _s.combine_many_boxes([line_info.get_box(j) for j in cur_ind])
        flb.append((x,y,w,h))
        self.widths.append(w*scale)
        return w

    def _combine_line_boxes(self, line_info, line, scale):
        fli = [] # Final Line indices
        flb = []
        line_widths = []

        line = iter(line)

        # Initialize the current box and its attrs#         BREAKWIDTH = 3.0
        cur_ind = [next(line)]

        # cb is current box, b is the next box
        cb = line_info.get_box(cur_ind[0])

        cur_attrs = box_attrs(cb)

        # Loop through box, combine and close along the way
        for i in line:
            b = line_info.get_box(i)
            next_attrs = box_attrs(b)

            is_interior, is_tsheg, tsheg_separation = self._combine_flags(line_info, i, cur_attrs, next_attrs)

            # Separate when the next box starts past the group's right edge, on
            # tsheg_separation, or when the overlap chain says so; absorb otherwise.
            if next_attrs[0] > cur_attrs[1] or tsheg_separation or self._should_separate(is_interior, is_tsheg, cur_attrs, next_attrs, i):
                self._flush_group(line_info, fli, flb, cur_ind, scale)
                cur_ind = [i]
                cb = b
                cur_attrs = box_attrs(cb)
            else: # There is overlap: absorb into the current group
                cur_ind.append(i)
                bxs = [line_info.get_box(j) for j in cur_ind]
                bxs.append(b)
                cb = _s.combine_many_boxes(bxs)
                cur_attrs = box_attrs(cb)

        w = self._flush_group(line_info, fli, flb, cur_ind, scale)
        line_widths.append(w)
        return fli, flb, line_widths

    @staticmethod
    def _is_char_tsheg(line_info, char_idx, topn, bottomn):
        char_box = line_info.get_box(char_idx)
        by_width = in_tsek_pop(line_info.shapes, char_box[2], char_box[1], topn,
                               char_box[1]+char_box[3], bottomn, char_idx)
        by_small = hasattr(line_info.shapes, 'small_contour_indices') \
            and char_idx in line_info.shapes.small_contour_indices
        return by_width or by_small

    @staticmethod
    def _group_contains_tsheg(line_info, group, topn, bottomn):
        # Check if any character in this group is a tsheg - if so, don't merge
        for char_idx in group:
            if CombineBoxesForPage._is_char_tsheg(line_info, char_idx, topn, bottomn):
                _s._dbg(f"[TSHEG PROTECTION] Preventing low-ink re-merging of tsheg at index {char_idx}")
                return True
        return False

    def _low_ink_target(self, line_info, lib, box, group):
        """Index of the low-ink box `group` should re-merge into, or None to keep
        it separate (no enclosing low-ink box, or the group contains a tsheg).

        ###  This attempts to remove noise that doesn't fall into a blurred
        ### low-ink box but does get combined according to normal combination rules
        """
        led, red = box_attrs(box)[:2]
        for p, bx in enumerate(lib):
            ledn, ren, topn, bottomn = box_attrs(bx)[:4]
            if led >= ledn-15 and red <= ren+15:
                if self._group_contains_tsheg(line_info, group, topn, bottomn):
                    return None   # keep separate to preserve the tsheg
                return p
        return None

    def _low_ink_regroup(self, line_info, lineind, fli, flb, scale):
        lib = self.line_info.low_ink_boxes[lineind]
        low_ink_segmentation = {}
        not_intr = []
        for d, box in enumerate(flb):
            p = self._low_ink_target(line_info, lib, box, fli[d])
            if p is None:
                not_intr.append(fli[d])
            else:
                low_ink_segmentation.setdefault(p, []).extend(fli[d])

        all_li_seg = list(low_ink_segmentation.values())
        all_li_seg.extend(not_intr)
        newfli = []
        newflb = []
        for j in all_li_seg:
            newfli.append(j)
            x,y,w,h = _s.combine_many_boxes([line_info.get_box(i) for i in j])
            newflb.append([x,y,w,h])
            self.widths.append(w*scale)

        fliflb = list(zip(newfli, newflb))
        fliflb.sort(key=lambda x: x[1][0])

        fli = [i[0] for i in fliflb]
        flb = [i[1] for i in fliflb]
        return fli, flb

    def combine_for_line_ind(self, line_info, lineind=None):
        line = line_info.lines_chars[lineind]

        line.sort(key=lambda x: line_info.get_box(x)[0])

        if not line:
            return []

        scale = self._line_scale(line_info, lineind, line)
        self.transitions.append(scale)

        fli, flb, line_widths = self._combine_line_boxes(line_info, line, scale)

        if line_info.shapes.low_ink:
            fli, flb = self._low_ink_regroup(line_info, lineind, fli, flb, scale)

        self.final_indices.append(fli)
        self.final_boxes.append(flb)
        self.line_width_means.append(mean(line_widths))
