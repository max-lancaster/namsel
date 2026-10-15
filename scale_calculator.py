#! /usr/bin/python
# encoding: utf-8
'''ScaleCalculator - Scale-adaptive parameter computation, extracted from page_elements2.py'''

import cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

try:
    from .config_manager import default_config
    from .utils import adaptive_binary_inv
except ImportError:
    from config_manager import default_config
    from utils import adaptive_binary_inv


class ScaleCalculator(object):
    """Mixin class providing scale calculation methods for PageElements."""

    @classmethod
    def char_gaussians(cls, widths):

        widths = np.array(widths)
        widths.shape = (len(widths),1)
        cls.median_width = np.median(widths)

        gmm = GMM(n_components = 2, max_iter=100, random_state=42)
        try:
            gmm.fit(widths)
        except ValueError:
            return (0,0,0,0)
        means = gmm.means_
        # Handle different scikit-learn versions
        stds = np.sqrt(gmm.covariances_) if hasattr(gmm, 'covariances_') else np.sqrt(gmm.covars_)
        cls.gmm = gmm
        char_mean_ind = np.argmax(means)
        # numpy 2.x compatible: use .flat[0] instead of direct float() on array
        char_mean = float(np.asarray(means[char_mean_ind]).flat[0]) # Page character width mean
        char_std = float(np.asarray(stds[char_mean_ind]).flat[0]) # Page character std dev

        cls.tsek_mean_ind = np.argmin(means)
        tsek_mean = float(np.asarray(means[cls.tsek_mean_ind]).flat[0])
        tsek_std = float(np.asarray(stds[cls.tsek_mean_ind]).flat[0])

        if cls._is_degenerate_gmm(char_std, tsek_mean, char_mean, len(widths)):
            return cls._median_based_widths(widths, float(cls.median_width))
        return (char_mean, char_std, tsek_mean, tsek_std)

    @staticmethod
    def _is_degenerate_gmm(char_std, tsek_mean, char_mean, n_widths):
        """A GMM split is unreliable when char_std collapses on a tiny sample (a few
        wide diacritics land in the "char" class), or the tsek-class mean sits
        suspiciously close to the char-class mean (real tsheks are ~1/3–1/4 as wide)."""
        if char_std < 0.5 and n_widths < 30:
            return True
        return tsek_mean > 0 and char_mean > 0 and tsek_mean / char_mean > 0.6

    @staticmethod
    def _median_based_widths(widths, median_w):
        """Fallback char/tsek stats by splitting widths around half the median."""
        flat_widths = widths.flatten()
        small = flat_widths[flat_widths < median_w * 0.5]
        large = flat_widths[flat_widths >= median_w * 0.5]
        if len(large) >= 2:
            char_mean, char_std = float(np.mean(large)), float(np.std(large))
        else:
            char_mean, char_std = median_w, 0.0
        if len(small) >= 1:
            tsek_mean, tsek_std = float(np.mean(small)), float(np.std(small))
        else:
            tsek_mean, tsek_std = char_mean / 3.0, 0.0  # no clear tsheks → ~1/3 char
        return (char_mean, char_std, tsek_mean, tsek_std)

    @staticmethod
    def _collect_char_widths(contours):
        """Widths of character-like components (reasonable size, filtering out noise
        and page frames)."""
        widths = []
        for contour in contours:
            _, _, w, h = cv.boundingRect(contour)
            if w > 5 and h > 5 and w < 200 and h < 200:
                widths.append(w)
        return widths

    def _calculate_preliminary_scale_factor(self):
        """
        Calculate a preliminary scale factor based on image characteristics
        before full contour analysis. This enables proper scale-adaptive
        thresholding from the start.
        """
        # Bootstrap scale from image height (paragraph2.png 106px works at scale 1.0),
        # then derive the same scale-adaptive odd block size + C as the main pipeline.
        # (img height is conversion-invariant, so self.img_arr.shape[0] == the old
        # img_copy.shape[0].)
        bootstrap_scale = max(0.5, min(3.0, self.img_arr.shape[0] / 106))
        self.bootstrap_scale = bootstrap_scale
        basic_binary = adaptive_binary_inv(self.img_arr, bootstrap_scale, getattr(self, 'force_single_line', False))
        result = cv.findContours(basic_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = result[0] if len(result) == 2 else result[1]

        if not contours:
            self.preliminary_scale_factor = 1.0
            return
        self._preliminary_contour_count = len(contours)

        widths = self._collect_char_widths(contours)
        if not widths:
            self.preliminary_scale_factor = 1.0
            return

        # Scale = median char width vs the paragraph2.png reference (14.23), clamped.
        self.preliminary_scale_factor = max(0.3, min(25.0, np.median(widths) / 14.23))

    def _calculate_adaptive_bounds(self):
        """
        Calculate scale-adaptive bounds for the entire OCR pipeline based on actual Tibetan text scale.
        Key insight: All processing parameters should scale with text size, not just tsheg detection.
        Returns bounds and global scale factor for use throughout the pipeline.
        """
        if not hasattr(self, "img_arr") or self.img_arr is None:
            # Fallback to fixed bounds if no image
            return {
                "min_area": 9, "max_area": 25,
                "min_size": 3, "max_size": 6,
                "scale_factor": 1.0
            }

        # Use preliminary_scale_factor if available (calculated from character width estimation)
        if hasattr(self, 'preliminary_scale_factor'):
            scale_factor = self.preliminary_scale_factor
            if default_config.get('debug_output', False):
                print(f"[ADAPTIVE BOUNDS] Using preliminary_scale_factor: {scale_factor:.3f}")
        else:
            # Fallback to character height analysis
            scale_factor = self._calculate_scale_from_char_height()
            if default_config.get('debug_output', False):
                print(f"[ADAPTIVE BOUNDS] Using char height scale: {scale_factor:.3f}")

        # Calculate adaptive bounds using the scale factor
        ref_min_size, ref_max_size = 3, 6
        ref_min_area, ref_max_area = 9, 25

        min_size = max(1, int(ref_min_size * scale_factor))
        max_size = max(min_size + 1, int(ref_max_size * scale_factor))
        min_area = max(1, int(ref_min_area * scale_factor * scale_factor))
        max_area = max(min_area + 1, int(ref_max_area * scale_factor * scale_factor))

        bounds = {
            "min_area": min_area, "max_area": max_area,
            "min_size": min_size, "max_size": max_size,
            "scale_factor": scale_factor
        }

        if default_config.get('debug_output', False):
            print(f"[ADAPTIVE BOUNDS] Calculated bounds: area={min_area}-{max_area}, size={min_size}-{max_size}, scale={scale_factor:.3f}")
        return bounds

    def _calculate_scale_from_char_height(self):
        """Calculate scale factor based on character height analysis"""
        # Get contour boxes for character height measurement
        try:
            contour_boxes = self.get_boxes()
        except:
            # Fallback if contours not ready yet
            return 1.0

        tibetan_heights = self._measure_tibetan_heights(contour_boxes)

        if tibetan_heights:
            avg_char_height = np.median(tibetan_heights)
        else:
            # Fallback: estimate from image height (~15 lines).
            avg_char_height = max(10, min(50, self.img_arr.shape[0] // 15))

        # Reference character height from paragraph.png analysis (20 px), clamped 0.3–5.0.
        return max(0.3, min(5.0, avg_char_height / 20))

    @staticmethod
    def _measure_tibetan_heights(contour_boxes):
        """Heights of contours passing the character size heuristic (reasonable
        size + Tibetan-char height/width band)."""
        ref = {'min_height': 5, 'max_height': 50, 'min_width': 3, 'max_width': 40,
               'min_area': 15, 'max_area': 2000, 'char_min_height': 10, 'char_max_height': 35,
               'char_min_width': 5, 'char_max_width': 25}
        heights = []
        for (x, y, w, h) in contour_boxes:
            area = w * h
            if not (ref['min_height'] <= h <= ref['max_height'] and
                    ref['min_width'] <= w <= ref['max_width'] and
                    ref['min_area'] <= area <= ref['max_area']):
                continue
            if (ref['char_min_height'] <= h <= ref['char_max_height'] and
                    ref['char_min_width'] <= w <= ref['char_max_width']):
                heights.append(h)
        return heights

    def _get_scaled_params(self, scale_factor=None):
        """
        Get scale-adaptive parameters for all preprocessing operations.
        Converts all hardcoded pixel values to scale-adaptive values.
        """
        if scale_factor is None:
            scale_factor = getattr(self, 'global_scale_factor', 1.0)

        # Reference parameters (based on paragraph.png which works at scale=1.0)
        params = {
            # Adaptive threshold parameters
            'adaptive_block_size': max(3, int(11 * scale_factor)),
            'adaptive_c_param': max(1, int(2 * scale_factor)),

            # Morphological operation parameters
            'erosion_kernel_horizontal': max(1, int(2 * scale_factor)),
            'erosion_kernel_vertical': max(1, int(1 * scale_factor)),
            'erosion_iterations': max(1, int(2 * scale_factor)),
            'blur_horizontal': int(75 * scale_factor),
            'blur_vertical': int(19 * scale_factor),

            # Contour filtering parameters
            'min_contour_area': int(15 * scale_factor * scale_factor),  # Area scaling
            'max_contour_area': int(2000 * scale_factor * scale_factor),  # Area scaling
            'min_contour_width': max(1, int(3 * scale_factor)),
            'max_contour_width': int(40 * scale_factor),
            'min_contour_height': max(1, int(5 * scale_factor)),
            'max_contour_height': int(50 * scale_factor),

            # Character size filtering parameters
            'char_min_width': max(1, int(5 * scale_factor)),
            'char_max_width': int(25 * scale_factor),
            'char_min_height': max(1, int(10 * scale_factor)),
            'char_max_height': int(35 * scale_factor),

            # Distance and overlap thresholds
            'distance_threshold': 1.5 * scale_factor,
            'narrow_threshold': 1.0 * scale_factor,
            'component_split_area': int(400 * scale_factor * scale_factor),

            # Y-range margins for position filtering
            'y_range_margin': int(10 * scale_factor),
            'multiline_threshold': int(50 * scale_factor),

            # Syllable merging parameters
            'merge_horizontal_distance_multiplier': 1.5,  # Ratio - no scaling
            'merge_vertical_proximity_multiplier': 0.8,   # Ratio - no scaling
            'merge_area_ratio': 0.3,  # Ratio - no scaling

            # Global scale factor for reference
            'scale_factor': scale_factor
        }

        return params
