#! /usr/bin/python
# encoding: utf-8
'''Page Elements - Refactored into modular components.

Core PageElements class with __init__, filtering logic, and coordination.
Heavy lifting delegated to mixin classes:
  - ScaleCalculator: GMM analysis, scale factor computation
  - ContourProcessor: contour detection, merging, drawing
  - LayoutDetector: pecha layout, line detection, margin content
'''

import cv2 as cv
import numpy as np
from scipy.stats import mode as statsmode
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class PageElementsOptions:
    """Optional configuration for PageElements' first-pass segmentation.

    Bundled into one object so PageElements.__init__ stays within the parameter
    limit; every field keeps the historical default of the old keyword argument.
    """
    small_coef: int = 1          # lower coef means more filtering; USE 3 for nying gyud
    low_ink: bool = False
    page_type: str = None
    flpath: str = None
    detect_o: bool = True
    clear_hr: bool = False
    force_single_line: bool = False

try:
    from .classify import load_cls
    from .config_manager import default_config
except ImportError:
    from classify import load_cls
    from config_manager import default_config

# Load the classifier as fast_cls for compatibility
fast_cls = load_cls('logistic-cls')

# Import ML-based tsheg separator (will be imported dynamically to avoid circular imports)
ML_AVAILABLE = False

# Import extracted modules (dual import pattern for package/standalone usage)
try:
    from .scale_calculator import ScaleCalculator
    from .contour_processor import ContourProcessor
    from .layout_detector import LayoutDetector
except ImportError:
    from scale_calculator import ScaleCalculator
    from contour_processor import ContourProcessor
    from layout_detector import LayoutDetector



class PageElements(ScaleCalculator, ContourProcessor, LayoutDetector):
    '''Page Elements object - a representation of the tiff image as a set
    of elements (contours, bounding boxes) and measurements used for recognition

    Parameters:
    -----------
    img_arr: 2d numpy array containing pixel data of the image

    small_coef: int, default=2
        A scalar value used in filtering out small ("noise") objects in the
        image.

        This may be deprecated soon. It is useful in situations where you
        know the typeset being used and want to ensure filtering is not too
        lax or aggressive.

    Attributes:
    ------
    contours: list, a list of contours return by cv.findContours

    hierarchy: list, contour hierarchy exported by cv.findContours

    boxes: list, list of bounding boxes for the page

    indices: list, list of integers representing the indices for contours and
        boxes that have not been filtered

    char_mean, char_std, tsek_mean, tsek_std: float, parameters of the Gaussian
        distributions for letters and punctuation on the page (first pass)

    page_array: 2d array of containing newly drawn image with filtered blobs
        removed

    Methods:
    --------
    char_gaussians: class method for using 2 class GMM

    get_tops: helper function for getting the top y coordinates of all
        bounding boxes on the page (-filter boxes)
    '''


#     @timeout(25)
#     @profile
    def __init__(self, img_arr, fast_cls, opts=None):
        if opts is None:
            opts = PageElementsOptions()
        self.img_arr = img_arr
        self.page_type = opts.page_type
        self.flpath = opts.flpath
        self.low_ink = opts.low_ink
        self.detect_o = opts.detect_o
        self.force_single_line = opts.force_single_line
        self.cached_features = OrderedDict()
        self.cached_pred_prob = OrderedDict()
        # FIXED: Use RETR_EXTERNAL instead of RETR_TREE to match preliminary stage
        # RETR_TREE was causing over-segmentation by finding internal contours within compound syllables
        # RETR_EXTERNAL only finds outermost contours, preventing syllable splitting
        self._contour_mode = cv.RETR_TREE

        self._calculate_preliminary_scale_factor()
        self.adaptive_bounds = self._calculate_adaptive_bounds()

        # OpenCV version-agnostic contour unpacking (3.x 3-tuple / 4.x 2-tuple).
        contour_result = self._contours()
        if len(contour_result) == 2:
            self.contours, self.hierarchy = contour_result
        else:
            _, self.contours, self.hierarchy = contour_result

        self.boxes = []
        self.indices = []
        self.small_coef = opts.small_coef
        self.warning_count = 0  # guard against runaway processing

        self._set_shape_measurements()

        # Default adaptive filtering ranges (overwritten by _analyze_document_type).
        self.adaptive_y_min = 5
        self.adaptive_y_max = 35
        self.is_multiline = False
        if hasattr(self, 'contours') and self.contours:
            self._analyze_document_type(self.contours)

        content_parent = self._compute_content_parent(opts.page_type, opts.clear_hr)
        outer_contours, outer_widths = self._select_outer_contours(content_parent, opts.clear_hr)
        self._set_width_measures(outer_widths)

        self.small_contour_indices = []
        self.indices = []   # reset — rebuilt by the classification pass
        self.emph_symbols = []
        self.naros = []
        self._classify_outer_contours(outer_contours)

        if self.detect_o:
            print(('pre-filtered na-ro vowel', len(self.naros), 'found'))
        if self.low_ink:
            self._low_ink_setting()

    def _compute_content_parent(self, page_type, clear_hr):
        """Pick the dominant parent-contour id (the text body) and populate
        self.indices; runs pecha layout / hr-clearing when applicable."""
        if page_type == 'pecha':
            if clear_hr:
                print('Warning: clear_hr called on pecha format. For clearing text')
                self.force_clear_hr()
            self.set_pecha_layout()
            if self.indices:
                return int(statsmode([self.hierarchy[0][i][3] for i in self.indices])[0])
            print('no content found')
            return None
        self.indices = self.get_indices()
        if not self._has_valid_hierarchy():
            return -1
        return int(statsmode([hier[3] for hier in self.hierarchy[0]])[0])

    def _has_valid_hierarchy(self):
        return not (self.hierarchy is None or len(self.hierarchy) == 0 or self.hierarchy[0] is None)

    def _select_outer_contours(self, content_parent, clear_hr):
        """Collect the outer (content-parent) contours + their widths for the GMM
        char-size estimate; skip/clear aesthetic hr lines along the way."""
        outer_contours = []
        outer_widths = []
        img_w = self.img_arr.shape[1]
        for i in self.indices:
            cbox = self.get_boxes()[i]
            # Never count contours wider than 30% of the page as characters.
            if self.hierarchy[0][i][3] == content_parent and cbox[2] < 0.3 * img_w:
                outer_contours.append(i)
                outer_widths.append(cbox[2])
            elif cbox[2] <= 0.3 * img_w:
                self._maybe_clear_hr(cbox, clear_hr, img_w)
        return outer_contours, outer_widths

    def _maybe_clear_hr(self, cbox, clear_hr, img_w):
        """Clear a horizontal rule (wide bar near the top) from the page array."""
        if cbox[2] > .66 * img_w:
            print((cbox[2] / float(img_w)))
        if clear_hr and .995 * img_w > cbox[2] > .66 * img_w and cbox[1] < .25 * self.img_arr.shape[0]:
            self.img_arr[0:cbox[1] + cbox[3], :] = 1

    def _set_width_measures(self, outer_widths):
        """GMM char/tsek width stats + the global scale factor, then recompute the
        adaptive bounds with that scale."""
        width_measures = self.char_gaussians(outer_widths)
        for name, val in zip(['char_mean', 'char_std', 'tsek_mean', 'tsek_std'], width_measures):
            setattr(self, name, val)
        raw_scale_factor = self.char_mean / 14.23  # paragraph2.png reference char_mean
        self.global_scale_factor = max(0.3, min(25.0, raw_scale_factor))
        self.adaptive_bounds = self._calculate_adaptive_bounds()

    def _classify_outer_contours(self, outer_contours):
        """Per outer contour: skip tiny shapes, run the punctuation heuristic, and
        route it to the small/main pipelines."""
        try:
            from .classify import label_chars
        except ImportError:
            from classify import label_chars
        num_classes = len(label_chars)
        for i in outer_contours:
            if self.warning_count > 1000:
                print("[WARNING] Too many processing warnings, stopping character recognition")
                break
            cbox = self.get_boxes()[i]
            x, y, w, h = cbox
            scale_factor = self._get_scaled_params()['scale_factor']
            min_wh = max(1, int(scale_factor))
            if w < min_wh or h < min_wh:
                continue
            self.draw_contour_and_children(i, np.ones((h, w), dtype=np.uint8), (-x, -y))
            area = w * h
            aspect_ratio = h / w if w > 0 else 1.0
            prprob = self._heuristic_prprob(w, h, area, aspect_ratio, scale_factor, num_classes)
            mxinx = prprob.argmax()
            quick_prd = label_chars.get(mxinx, '?')
            self.cached_pred_prob[i] = (mxinx, prprob[0])
            self._classify_small_or_main(i, cbox, quick_prd)

    @staticmethod
    def _heuristic_prprob(w, h, area, aspect_ratio, scale_factor, num_classes):
        """Quick size/shape heuristic → a class-probability row (tsheg 510 / shad 511 /
        neutral) used only for early punctuation routing; segment.py does the real
        classification for everything else."""
        prprob = np.zeros((1, num_classes))
        # Areas grow with the square of the scale; a limit linear in it rejected 12x13 tshegs at scale 3.
        area_scale = scale_factor * max(1.0, scale_factor)
        if area <= int(50 * area_scale) and w <= int(15 * scale_factor) and h <= int(15 * scale_factor):
            cls_ = 510  # ་ tsheg (small square-ish)
        elif area <= int(350 * area_scale) and aspect_ratio >= 4.0 and w <= int(10 * scale_factor):
            cls_ = 511  # ། shad (tall thin)
        else:
            prprob[0][0] = 1.0  # neutral — let segment.py classify
            return prprob
        if cls_ < num_classes:
            prprob[0][cls_] = 0.8
            prprob[0][0] = 0.2
        else:
            prprob[0][0] = 1.0
        return prprob

    def _adaptive_y_range(self):
        """(y_min, y_max) text-line band for punctuation filtering; a scale-adaptive
        fallback when _analyze_document_type hasn't set adaptive_y_min yet."""
        if not hasattr(self, 'adaptive_y_min'):
            sp = self._get_scaled_params()
            y_min = max(1, int(5 * sp['scale_factor']))
            return y_min, max(y_min + 10, int(35 * sp['scale_factor']))
        return self.adaptive_y_min, self.adaptive_y_max

    def _classify_small_or_main(self, i, cbox, quick_prd):
        """Route a contour: small components get punctuation-routed or filtered;
        larger components go to the main character indices."""
        scale_factor = self._get_scaled_params()['scale_factor']
        if scale_factor > 2.5:
            threshold = max(15, int(6 * scale_factor))
        elif scale_factor > 1.5:
            threshold = max(10, int(8 * scale_factor))
        else:
            threshold = max(7, int(7 * scale_factor))
        if cbox[2] < threshold:
            self._route_small_component(i, cbox, quick_prd)
        else:
            self.indices.append(i)

    def _route_small_component(self, i, cbox, quick_prd):
        """Small component: drop vowel-marks/noise outside the text band or too-tiny
        blobs; route tsheg/shad to their pipelines; drop everything else."""
        y = cbox[1]
        area = cbox[2] * cbox[3]
        y_min, y_max = self._adaptive_y_range()
        if not (y_min <= y <= y_max):
            return  # vowel mark / noise outside the text line
        if area < 9:
            return  # too tiny
        if quick_prd == '་':
            self._route_tsheg(i, cbox)
        elif quick_prd == '།':
            self._route_shad(i, cbox)
        # else: not Tibetan punctuation and too small → drop

    def _route_tsheg(self, i, cbox):
        """A ་-classified small component: position-filter, then statistical (or
        fixed-threshold) size validation before adding to a pipeline."""
        area = cbox[2] * cbox[3]
        y = cbox[1]
        y_min, y_max = self._adaptive_y_range()
        if not (y_min <= y <= y_max):
            print(f"[PUNCT DEBUG] FILTERED tsheg at i={i}, box={cbox} [REASON: y={y} outside adaptive range {y_min}-{y_max}]")
            return
        if hasattr(self, 'tsek_mean') and self.tsek_mean > 0:
            self._route_tsheg_statistical(i, cbox, area)
        else:
            self._route_tsheg_fixed(i, cbox, area)

    def _route_tsheg_statistical(self, i, cbox, area):
        """Scale-adaptive statistical tsheg size gate → small_contour_indices."""
        scale_factor = getattr(self, 'global_scale_factor', 1.0)
        min_width = max(2, int(3 * scale_factor))
        max_width_base = int(6 * scale_factor)
        min_area = max(6, int(7 * scale_factor * scale_factor))
        max_area_base = int(25 * scale_factor * scale_factor)
        if scale_factor > 2.0:
            max_width = max(max_width_base, int(12 * scale_factor))
            max_area = max(max_area_base, int(50 * scale_factor))
        else:
            max_width = max_width_base
            max_area = max_area_base
        tall_enough = cbox[3] >= min_width or self._flat_tsheg(cbox)
        if (cbox[2] >= min_width and tall_enough and
                cbox[2] <= max_width and cbox[3] <= max_width and
                min_area <= area <= max_area):
            self.small_contour_indices.append(i)
        # else: filtered (drop)

    def _flat_tsheg(self, cbox):
        """A tsheg a little flatter than the height minimum (Monlam Bodyig: 12x11 where the minimum is 12) is still
        a tsheg when it has this line's own tsheg width, is roughly square and solid, and is not cut by the band's edge
        (pieces of the neighbouring line and specks are not)."""
        x, y, w, h = cbox
        scale_factor = getattr(self, 'global_scale_factor', 1.0)
        if h < max(2, int(2 * scale_factor)) or w > 1.6 * h:
            return False
        if abs(w - self.tsek_mean) > max(1.0, self.tsek_std):
            return False
        if y <= 0 or y + h >= self.img_arr.shape[0]:
            return False
        box = self.img_arr[y:y + h, x:x + w]
        return box.size > 0 and float((box < 0.5).mean()) >= 0.35

    def _route_tsheg_fixed(self, i, cbox, area):
        """Fixed-threshold tsheg gate (no tsek_mean yet) → main indices."""
        if (cbox[2] >= 3 and cbox[3] >= 3 and cbox[2] <= 6 and cbox[3] <= 6 and 12 <= area <= 25):
            self.indices.append(i)
        # else: drop

    def _route_shad(self, i, cbox):
        """A །-classified small component → small_contour_indices (unless too short
        for a large-scale image)."""
        scale_factor = getattr(self, 'global_scale_factor', 1.0)
        if scale_factor > 2.0 and cbox[3] < 30:
            return  # too small for this scale
        self.small_contour_indices.append(i)

    def force_clear_hr(self):
        boxes = self.get_boxes()
        for cbox in boxes:
            if .995*self.img_arr.shape[1] > cbox[2] > \
                    .66*self.img_arr.shape[1] and cbox[1] < .25*self.img_arr.shape[0]:
                        self.img_arr[0:cbox[1]+cbox[3], :] = 1

    def _low_ink_setting(self):
        print('IMPORTANT: Low ink setting=True')
        a = self.img_arr.copy()*255

        erode_iter = 2
        vertblur = 35
        horizblur = 1
        threshold = 160

        # EXPERIMENTAL: Disable erosion to preserve syllable integrity
        if default_config.get('debug_output', False):
            print(f"[EROSION DEBUG] Skipping erosion (iterations={erode_iter}) to preserve syllables")
        a = cv.blur(a, (horizblur,vertblur))
        ret, a = cv.threshold(a, threshold, 255, cv.THRESH_BINARY)
        # OpenCV version-agnostic approach
        contours_result = cv.findContours(a, mode=self._contour_mode ,
                                         method=cv.CHAIN_APPROX_SIMPLE)
        ctrs, hier = contours_result[-2:]  # Get last 2 values regardless of OpenCV version

        self.low_ink_boxes = [cv.boundingRect(c) for c in ctrs]
        self.low_ink_boxes = [i for i in self.low_ink_boxes if
                              i[2] < 1.33*self.char_mean]
        del a, ctrs, hier

    def get_boxes(self):
        '''Retrieve bounding boxes. Create them if not yet cached'''
        if not self.boxes:
            self.boxes = self._boxes()

        return self.boxes

    def _boxes(self):
        return [cv.boundingRect(c) for c in self.contours]

    def get_indices(self):
        if not self.indices:
            self.indices = [i for i, b in enumerate(self.get_boxes())]
        return self.indices

    def _set_shape_measurements(self):
        width_measures = self.char_gaussians([b[2] for b in self.get_boxes() if
                                               b[2] < .1*self.img_arr.shape[1]])
        for i,j in zip(['char_mean', 'char_std', 'tsek_mean', 'tsek_std'], width_measures):
            setattr(self, i, j)
