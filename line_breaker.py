# encoding: utf-8
'''Line breaking'''
from numpy import  array, float64, argmax, argmin, uint8, ones, mean, std, where, argsort
import cv2 as cv
try:
    from .utils import  check_for_overlap
    from .fast_utils import ftrim, fadd_padding
except ImportError:
    from utils import  check_for_overlap
    from fast_utils import ftrim, fadd_padding
import sys
from bisect import bisect, bisect_right
try:
    from .feature_extraction import normalize_and_extract_features
    from .classify import load_cls, label_chars
    from .config_manager import default_config
except ImportError:
    from feature_extraction import normalize_and_extract_features
    from classify import load_cls, label_chars
    from config_manager import default_config

cls = load_cls('logistic-cls')

class LineCut(object):
    '''Line Cutting object - breaks lines in a page where lines are separated
    by empty whitespace
    
    Parameters:
    --------------------
    shapes: page_element object, (see page_elements.py)
    
    thresh_scale: float, default=.9995
        A threshold value for determining the breakline in the event that
        there is black pixel noise between lines. Should be set high to avoid
        setting line breaks through characters themselves. 
    
    Attributes:
    -----------
    lines_chars: list of lists, length=number of lines on page. Each sub-list
        contains the indices for the bounding boxes/contours assigned to
        its corresponding line.
    
    line_indices: list of int, indices of breaklines with respect to page_array
    
    baselines: list of int, the index of the baseline for each line where
        baseline here is usually a line that goes through all the thick "head"
        (Tibetan: mgo) parts found on most Tibetan letters
    
    Methods:
    --------
    
    get_box: return the bounding box for a given index
    get_contour: return the contour for a given index
    
    The get_box, get_contour methods are mostly here for API compatibility with
    LineCluster.
    '''

    @staticmethod
    def _find_line_indices(vsum, threshold):
        """Row indices where the ink-sum falls below threshold = line boundaries."""
        indices = []
        for i, s in enumerate(vsum):
            if s < threshold and vsum[i - 1] >= threshold:
                indices.append(i - 1)
        return indices

    @staticmethod
    def _find_too_tall(diffs):
        """Line gaps that are outliers (> 2σ + mean of the others) — too-tall lines."""
        too_tall = []
        for i in range(len(diffs)):
            otherdiffs = diffs[:i] + diffs[i + 1:]
            if diffs[i] > 2 * std(otherdiffs) + mean(otherdiffs):
                too_tall.append(i)
        return too_tall

    def __init__(self, shapes, thresh_scale=.9995):
        self.shapes = shapes
        self.baselines = []

        # Inflate chars (erode) so vowels don't break off their lines.
        inflated = cv.erode(shapes.img_arr.copy(), None, iterations=shapes.conf['line_cut_inflation'])
        self.vsum = inflated.sum(axis=1)

        vsum_max = self.vsum.max()
        self.line_indices = self._find_line_indices(self.vsum, vsum_max * thresh_scale)
        li = self.line_indices
        self.k = len(self.line_indices)

        # Line heights; bail (if configured) when any line is too tall to line-cut.
        diffs = [li[i+1] - li[i] - len(where(self.vsum[li[i]:li[i+1]] == vsum_max)[0]) for i in range(len(li[:-2]))]
        too_tall = self._find_too_tall(diffs)
        if shapes.conf['stop_line_cut'] and len(too_tall) > 0:
            return

        self.assign_char_indices()
        
    def get_baselines(self):
        
        if not self.baselines:
            for i, k in enumerate(self.line_indices[:-1]):
                vsum_vals = self.vsum[k:self.line_indices[i+1]]
                if vsum_vals.any():
                    self.baselines.append(k +\
                        argmin(vsum_vals))
            self.baselines.append(self.line_indices[-1]+\
                    argmin(self.vsum[self.line_indices[-1]:]))

        return self.baselines
            
    def _assign_main_lines(self, sorted_indices, char_tops):
        """Assign main-char indices to lines. Returns True if the caller should
        return early (single line, no breaks detected)."""
        if self.shapes.conf.get('force_single_line'):
            self.lines_chars = [sorted_indices]
            self.k = 1
            return False
        insert_idxs = [bisect(char_tops, (i - 1,)) for i in self.line_indices]
        self.lines_chars = []
        if not insert_idxs:
            # No line breaks found — treat the whole page as a single line.
            self.lines_chars = [sorted_indices]
            self.k = 1
            return True
        for i in range(len(insert_idxs) - 1):
            self.lines_chars.append(sorted_indices[insert_idxs[i]:insert_idxs[i + 1]])
        self.lines_chars.append(sorted_indices[insert_idxs[-1]:])
        self.k = len(self.line_indices)
        return False

    def _main_line_y_coords(self):
        """Average y of each main line (inf for empty lines) for small-char proximity."""
        boxes = self.shapes.get_boxes()
        coords = []
        for line_chars in self.lines_chars:
            if line_chars:
                ys = [boxes[idx][1] for idx in line_chars]
                coords.append(sum(ys) / len(ys))
            else:
                coords.append(float('inf'))
        return coords

    @staticmethod
    def _closest_line(y_coord, main_line_y_coords):
        best_line = 0
        min_distance = float('inf')
        for line_idx, main_y in enumerate(main_line_y_coords):
            if main_y != float('inf') and abs(y_coord - main_y) < min_distance:
                min_distance = abs(y_coord - main_y)
                best_line = line_idx
        return best_line

    def _assign_small_to_nearest(self, char_tops, main_line_y_coords):
        for y_coord, char_idx in char_tops:
            best_line = self._closest_line(y_coord, main_line_y_coords)
            if best_line < len(self.small_cc_lines_chars):
                self.small_cc_lines_chars[best_line].append(char_idx)

    def _assign_small_contours(self):
        """Assign each small contour (tsheg/punctuation) to the nearest main line."""
        boxes = self.shapes.get_boxes()
        cctops = [boxes[i][1] for i in self.shapes.small_contour_indices]
        char_tops = sorted(zip(cctops, self.shapes.small_contour_indices), key=lambda x: x[0])
        sorted_indices = [i[1] for i in char_tops]
        self.small_cc_lines_chars = [[] for _ in range(len(self.lines_chars))]
        if self.shapes.conf.get('force_single_line') and self.small_cc_lines_chars:
            self.small_cc_lines_chars[0] = sorted_indices
        else:
            self._assign_small_to_nearest(char_tops, self._main_line_y_coords())
        # Filter out empty main lines (matches the original logic).
        self.small_cc_lines_chars = [self.small_cc_lines_chars[i]
                                     for i in range(len(self.lines_chars)) if self.lines_chars[i]]

    def _group_by_lines(self, cctops, items):
        """Sort (top, item) pairs, bisect at line_indices, split into per-line groups,
        then drop groups for empty main lines. Shared by naros + low-ink boxes."""
        char_tops = sorted(zip(cctops, items), key=lambda x: x[0])
        sorted_indices = [i[1] for i in char_tops]
        insert_idxs = [bisect_right(char_tops, (i - 1,)) for i in self.line_indices]
        if not insert_idxs:
            sys.exit()
        groups = [sorted_indices[insert_idxs[i]:insert_idxs[i + 1]] for i in range(len(insert_idxs) - 1)]
        groups.append(sorted_indices[insert_idxs[-1]:])
        return [groups[i] for i in range(len(self.lines_chars)) if self.lines_chars[i]]

    def assign_char_indices(self):
        '''
        Notes:
        ------
        Complexity:
        nlogn for sorting, linear for zip/index extraction, bisect is log n.
        '''
        char_tops = sorted(zip(self.shapes.get_tops(), self.shapes.get_indices()), key=lambda x: x[0])
        sorted_indices = [i[1] for i in char_tops]

        if self._assign_main_lines(sorted_indices, char_tops):
            return

        self._assign_small_contours()

        if self.shapes.detect_o:
            boxes = self.shapes.get_boxes()
            self.line_naros = self._group_by_lines(
                [boxes[i][1] for i in self.shapes.naros], self.shapes.naros)

        if self.shapes.low_ink:
            self.low_ink_boxes = self._group_by_lines(
                [lib[1] for lib in self.shapes.low_ink_boxes], self.shapes.low_ink_boxes)
        
#
#        for c in small_cc:
#            t = self.shapes.get_boxes()[c][1]
#            for i, line in enumerate(self.lines_chars):
#                if i+1 < len(self.lines_chars):
#                    if self.lines_chars[i+1]:
#                        next_topmost =  self.shapes.get_boxes()[self.lines_chars[i+1][0]][1]
#                        if topmost < t and t < next_topmost:
#                            self.small_cc_lines_chars[i].append(c)
#                            break
#                        topmost = next_topmost
#        
        
    def _remove_small_noise(self):
        global_tsek_mean = self.shapes.tsek_mean
        global_tsek_std = self.shapes.tsek_std
        global_char_mean = self.shapes.char_mean
        global_char_std = self.shapes.char_std
        for l in self.lines_chars:
            widths = [self.shapes.get_boxes()[i][2] for i in l]
            char_mean, char_std, tsek_mean, tsek_std = self.shapes.char_gaussians(widths)
            for i in l: 
                if self.shapes.get_boxes()[i][2] < (global_tsek_mean - 
                   1 * global_tsek_std) and not char_mean < global_char_mean -.5* global_char_std:
                    l.remove(i)
#                elif len(l) < 4 and global_tsek_mean - global_tsek_std <= self.shapes.get_boxes()[i][2] <= global_tsek_mean + global_tsek_std:
#                    l.remove(i)
        
    # These two functions are for API compatability with LineCluster
    def get_box(self, ind):
        return self.shapes.get_boxes()[ind]
    
    def get_contour(self, ind):
        return self.shapes.contours[ind]

class LineCluster(object):
    '''Line Cluster object - breaks lines by clustering according to tops of
    bounding boxes. Useful in cases where it drawing a straight line between
    page lines is difficult. Requires you know how many lines are on the page
    beforehand.
    
    Parameters:
    --------------------
    shapes: page_element object, (see page_elements.py)
    
    k: int, required
        number of lines on the page
    
    Attributes:
    -----------
    lines_chars: list of lists, length=number of lines on page. Each sub-list
        contains the indices for the bounding boxes/contours assigned to
        its corresponding line.
    
    line_indices: list of int, indices of breaklines with respect to page_array
    
    baselines: list of int, the index of the baseline for each line where
        baseline here is usually a line that goes through all the thick "head"
        (Tibetan: mgo) parts found on most Tibetan letters
    
    Methods:
    --------
    
    get_box: return the bounding box for a given index
    get_contour: return the contour for a given index
    
    The get_box, get_contour methods are mostly here for API compatibility with
    LineCluster.
    
    Notes:
    ------
    This code is messy and in progress. Some of the logic in the end contains
    hardcoded values particular to the Nyingma Gyudbum which is obviously 
    useless for general cases.
    '''
    def __init__(self, shapes, k):
        from sklearn.cluster import KMeans
        self.shapes = shapes
        self.k = k
        self.page_array = shapes.img_arr

        kmeans, lines, sort_inx, boxes = self._cluster_lines(shapes, k, KMeans)
        breaklines, topmosts = self._compute_breaklines(lines, boxes)
        self.baselines = self._compute_baselines(breaklines)
        self.lines_chars = self._split_lines_across_breaklines(shapes, lines, breaklines, topmosts, boxes)
        self._assign_small_contours(breaklines)
        self._assign_emph_symbols(kmeans, sort_inx, k)
        if self.shapes.detect_o:
            self._assign_naros(breaklines)
        if self.shapes.low_ink:
            self._assign_low_ink(breaklines)

    def _cluster_tops(self, shapes):
        """The (n,1) array of char-top y-coordinates to cluster, per line_cluster_pos."""
        if shapes.conf['line_cluster_pos'] == 'top':
            tops = array(shapes.get_tops(), dtype=float64)
        elif shapes.conf['line_cluster_pos'] == 'center':
            tops = array(
                 [t[1] + .5*shapes.char_mean for t in shapes.get_boxes() if t[3] > 2* shapes.tsek_mean],
                    dtype=float64
                         )
        else:
            raise ValueError("The line_cluster_pos argument must be either 'top' or 'center'")
        tops.shape = (len(tops), 1)
        return tops

    def _cluster_lines(self, shapes, k, KMeans):
        """KMeans-cluster the char tops into k lines, drop empty clusters, and sort
        lines top-to-bottom. Returns (kmeans, lines, sort_inx, boxes)."""
        tops = self._cluster_tops(shapes)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(tops)

        lines = [[] for i in range(k)]
        ind = shapes.get_indices()
        ### Assign char pointers (ind) to the appropriate line ###
        [lines[kmeans.predict([[shapes.get_boxes()[ind[i]][1]]])[0]].append(ind[i]) for i in range(len(ind))]
        lines = [l for l in lines if l]
        self.k = len(lines)
        boxes = shapes.get_boxes()

        ### Sort indices so they are in order from top to bottom using y from the first box in each line
        sort_inx = list(argsort([boxes[line[0]][1] for line in lines]))
        lines.sort(key=lambda line: boxes[line[0]][1])
        return kmeans, lines, sort_inx, boxes

    def _compute_breaklines(self, lines, boxes):
        """Breaklines that split the page into line bands, from the topmost box in
        each cluster (snapped to the nearest ink peak). Returns (breaklines, topmosts)."""
        try:
            topmosts = [min([boxes[i][1] for i in line]) for line in lines]
        except ValueError:
            print('failed to get topmosts...')
            raise

        vsums = self.page_array.sum(axis=1)
        breaklines = []
        delta = 25
        for c in topmosts:
            if c - delta < 0:
                lower = 0
            else:
                lower = c-delta
            e = argmax(vsums[lower:c+delta])
            c = c - delta + e
            if c < 0:
                c = 0
            breaklines.append(c)

        breaklines.append(self.page_array.shape[0])
        return breaklines, topmosts

    def _compute_baselines(self, breaklines):
        """Per-line baseline = the min-ink row within each breakline band."""
        vsums = self.page_array.sum(axis=1)
        baselines = []
        for i, br in enumerate(breaklines[:-1]):
            try:
                baseline_area = vsums[br:breaklines[i+1]]
                if baseline_area.any():
                    baselines.append(br + argmin(baseline_area))
                else:
                    print(i)
                    print('No baseline info')
            except ValueError:
                print('ValueError. exiting...HERE')
                import traceback;traceback.print_exc()
                raise
        return baselines

    def _split_lines_across_breaklines(self, shapes, lines, breaklines, topmosts, boxes):
        """Assign each char to its line, splitting a char that extends over a
        breakline (a tall box crossing into the next line) into top/bottom pieces."""
        final_ind = dict((i, []) for i in range(len(lines)))
        self.new_contours = {}
        for j, br in enumerate(breaklines[1:-1]):
            topcount = 0
            bottomcount = 0
            for i in lines[j]:
                # A box/char must extend over the breakline by a non-trivial amount
                # (>= 30px) AND itself be tall-ish (~ a full line) to be broken.
                if (boxes[i][1] + boxes[i][3]) - br >= 30 and \
                    (boxes[i][1] + boxes[i][3]) - topmosts[j] > self.shapes.char_mean*2.85:
                    topcount, bottomcount = self._split_extending_char(
                        shapes, boxes, i, j, br, final_ind, topcount, bottomcount)
                else:
                    final_ind[j].append(i)
        # Don't forget to include the last line
        list(map(final_ind[len(lines)-1].append, lines[len(lines)-1]))
        return final_ind

    def _split_extending_char(self, shapes, boxes, i, j, br, final_ind, topcount, bottomcount):
        """Rasterize char i, find its top/bottom cut-point, and register the two
        resulting contours under 't{j}_{n}' / 'b{j}_{n}' names. Returns the updated
        (topcount, bottomcount)."""
        chars = ones((boxes[i][3]+2, boxes[i][2]+2), dtype=uint8)
        contours = shapes.contours
        cv.drawContours(chars, [contours[i]], -1, 0,
            thickness=-1, offset=(-boxes[i][0]+1, -boxes[i][1]+1))
        cv.dilate(chars, None, chars)
        y_offset = boxes[i][1]
        new_br = br - y_offset
        cut_point = self._find_char_cut_point(shapes, chars, new_br, br, boxes, i)
        c1, bnc1, c2, bnc2 = self._extract_split_contours(chars, cut_point, boxes, i)

        topbox_name = 't%d_%d' % (j, topcount)
        final_ind[j].append(topbox_name)
        self.new_contours[topbox_name] = (bnc1, c1)
        topcount += 1

        if bnc2[-1] > 8:  # only add bottom contour if not trivially small
            bottombox_name = 'b%d_%d' % (j, bottomcount)
            final_ind[j+1].append(bottombox_name)
            self.new_contours[bottombox_name] = (bnc2, c2)
            bottomcount += 1
        return topcount, bottomcount

    def _find_char_cut_point(self, shapes, chars, new_br, br, boxes, i):
        """Scan candidate cut-points around the breakline and pick the one whose
        top half classifies (non-tsek) with the highest probability."""
        prd_cut = []
        # Use scale-normalized tsek_mean (paragraph2.png has tsek_mean ~ 4.14).
        reference_tsek_mean = 4.14
        scale_factor = getattr(shapes, 'global_scale_factor', shapes.tsek_mean / reference_tsek_mean)
        normalized_tsek_mean = reference_tsek_mean * scale_factor

        for delta in range(-3, int(.75*normalized_tsek_mean), 1):
            cut_point = new_br + delta
            tchr = chars[:cut_point,:]
            tchr = ftrim(tchr)
            if not tchr.any():
                continue
            tchr = normalize_and_extract_features(tchr)
            # sklearn >= 1.0 requires a 2D (1, n_features) sample;
            # probs[0, ...] below already assumes the 2D result.
            probs = cls.predict_proba(tchr.reshape(1, -1))
            max_prob_ind = argmax(probs)
            chr = label_chars[max_prob_ind]
            prd_cut.append((probs[0,max_prob_ind], chr, cut_point))

        prd_cut = [q for q in prd_cut if q[1] != '་']
        try:
            return max(prd_cut)[-1]
        except:
            print('No max prob for vertical char break, using default breakline. Usually this means the top half of the attempted segmentation looks like a tsek blob')
            return br-boxes[i][1]

    def _extract_split_contours(self, chars, cut_point, boxes, i):
        """Contour + bounding box for the top and bottom halves of a char split at
        cut_point. Returns (c1, bnc1, c2, bnc2)."""
        tarr = chars[:cut_point,:]
        tarr, top_offset = ftrim(tarr, new_offset=True)
        tarr = fadd_padding(tarr, 3)
        barr = chars[cut_point:,:]
        barr = ftrim(barr, sides='brt')
        barr = fadd_padding(barr, 3)

        # Handle different OpenCV versions that return different numbers of values
        contour_result = cv.findContours(image=tarr, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE, offset=(boxes[i][0]+top_offset['left'],boxes[i][1]))
        if len(contour_result) == 2:
            c1, h = contour_result
        else:
            _, c1, h = contour_result

        c1 = c1[argmax([len(t) for t in c1])]  # use the most complex contour
        bnc1 = cv.boundingRect(c1)

        # Handle different OpenCV versions that return different numbers of values
        contour_result2 = cv.findContours(barr, mode=cv.RETR_TREE,
                                method=cv.CHAIN_APPROX_SIMPLE,
                                offset=(boxes[i][0]-3,boxes[i][1]+cut_point-3))
        if len(contour_result2) == 2:
            c2, h = contour_result2
        else:
            _, c2, h = contour_result2

        c2 = c2[argmax([len(t) for t in c2])]
        bnc2 = cv.boundingRect(c2)
        return c1, bnc1, c2, bnc2

    def _lines_from_breaklines(self, sorted_indices, char_tops, breaklines):
        """Split sorted_indices into per-breakline groups via bisect on char_tops."""
        _line_insert_indxs = [bisect_right(char_tops, (i - 1,)) for i in breaklines]
        if not _line_insert_indxs:
            sys.exit()
        grouped = []
        for i, l in enumerate(_line_insert_indxs[:-1]):
            grouped.append(sorted_indices[l:_line_insert_indxs[i+1]])
        grouped.append(sorted_indices[_line_insert_indxs[-1]:])
        return grouped

    def _keep_nonempty_lines(self, grouped):
        """Keep only the groups whose corresponding line has chars."""
        return [grouped[i] for i in range(len(self.lines_chars)) if self.lines_chars[i]]

    def _assign_small_contours(self, breaklines):
        """Distribute the small connected components across lines."""
        cctops = [self.shapes.get_boxes()[i][1] for i in self.shapes.small_contour_indices]
        char_tops = list(zip(cctops, self.shapes.small_contour_indices))
        char_tops.sort(key=lambda x: x[0])
        sorted_indices = [i[1] for i in char_tops]
        grouped = self._lines_from_breaklines(sorted_indices, char_tops, breaklines)
        self.small_cc_lines_chars = self._keep_nonempty_lines(grouped)

    def _assign_emph_symbols(self, kmeans, sort_inx, k):
        """Distribute emphasis symbols across lines by their KMeans cluster."""
        cctops = [self.shapes.get_boxes()[i][1] for i in self.shapes.emph_symbols]
        char_tops = list(zip(cctops, self.shapes.emph_symbols))
        char_tops.sort(key=lambda x: x[0])
        empred = [kmeans.predict(self.shapes.get_boxes()[i][1])[0] for i in self.shapes.emph_symbols]
        self.emph_lines = [[] for i in range(k)]
        for nn, e in enumerate(empred):
            self.emph_lines[sort_inx.index(e)].append(self.shapes.emph_symbols[nn])

    def _assign_naros(self, breaklines):
        """Distribute naro vowels across lines + record their box spans."""
        cctops = [self.shapes.get_boxes()[i][1] for i in self.shapes.naros]
        char_tops = list(zip(cctops, self.shapes.naros))
        char_tops.sort(key=lambda x: x[0])
        sorted_indices = [i[1] for i in char_tops]
        self.line_naros = self._keep_nonempty_lines(
            self._lines_from_breaklines(sorted_indices, char_tops, breaklines))
        self.line_naro_spans = []
        for ll, mm in enumerate(self.line_naros):
            thisline = []
            for nn, naro in enumerate(mm):
                box = self.get_box(naro)
                thisline.append(box)
            thisline.sort(key=lambda x: x[0])
            self.line_naros[ll].sort(key=lambda x: self.get_box(x)[0])
            self.line_naro_spans.append(thisline)

    def _assign_low_ink(self, breaklines):
        """Distribute low-ink boxes across lines."""
        cctops = [lib[1] for lib in self.shapes.low_ink_boxes]
        char_tops = list(zip(cctops, self.shapes.low_ink_boxes))
        char_tops.sort(key=lambda x: x[0])
        sorted_indices = [i[1] for i in char_tops]
        self.low_ink_boxes = self._keep_nonempty_lines(
            self._lines_from_breaklines(sorted_indices, char_tops, breaklines))

    def check_naro_overlap(self, line_num, box):
        line = self.line_naro_spans[line_num]
        left_edges = [l[0] for l in line]
        insert = bisect(left_edges, box[0])
        for r in range(insert-1, insert + 1):
            if 0 <= r < len(line):
                sp = line[r]
                if check_for_overlap(sp, box):
                    return self.line_naros[line_num][r]
        return False
        
    def get_box(self, ind):
        try:
            return self.shapes.get_boxes()[ind]
        except (TypeError, IndexError):
            return self.new_contours[ind][0]

    def get_contour(self, ind):
        try:
            return self.shapes.contours[ind]
        except (IndexError, TypeError):
            return self.new_contours[ind][1]
     
    def get_baselines(self):
        return self.baselines

