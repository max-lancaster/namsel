#! /usr/bin/python
# encoding: utf-8
'''ContourProcessor - Contour detection, merging, and drawing, extracted from page_elements2.py'''

import cv2 as cv
import numpy as np

try:
    from .config_manager import default_config
    from .utils import adaptive_binary_inv
except ImportError:
    from config_manager import default_config
    from utils import adaptive_binary_inv


class ContourProcessor(object):
    """Mixin class providing contour processing methods for PageElements."""

    @staticmethod
    def _unpack_findcontours(result):
        """OpenCV version-agnostic findContours unpacking → (contours, hierarchy)
        (3.x returns a 3-tuple, 4.x a 2-tuple)."""
        if len(result) == 2:
            return result
        return result[1], result[2]

    def _prep_binary(self):
        """Preprocess img_arr into an inverted adaptive-threshold binary. Uses a
        precise float→uint8 conversion (the old cast merged tshegs into main
        characters) and a scale-adaptive odd block size + C."""
        return adaptive_binary_inv(self.img_arr, getattr(self, 'bootstrap_scale', 1.0),
                                   getattr(self, 'force_single_line', False))

    def _filter_by_position(self, contours):
        """Keep contours whose top y is within the document's adaptive band."""
        filtered = []
        for contour in contours:
            _, y, _, _ = cv.boundingRect(contour)
            if self.adaptive_y_min <= y <= self.adaptive_y_max:
                filtered.append(contour)
        return filtered

    def _contours(self):
        bin_img = self._prep_binary()
        contours, hierarchy = self._unpack_findcontours(
            cv.findContours(bin_img, mode=self._contour_mode, method=cv.CHAIN_APPROX_SIMPLE))

        self.contour_index_mapping = None
        self._analyze_document_type(contours)   # sets adaptive_y_min/max
        filtered_contours = self._filter_by_position(contours)

        if len(filtered_contours) != len(contours):
            # Rebuild the hierarchy from only the kept contours.
            filtered_binary = np.zeros_like(bin_img)
            cv.drawContours(filtered_binary, filtered_contours, -1, 255, thickness=-1)
            return self._unpack_findcontours(
                cv.findContours(filtered_binary, mode=self._contour_mode, method=cv.CHAIN_APPROX_SIMPLE))

        return filtered_contours, hierarchy

    def _is_force_single_line(self):
        conf = getattr(self, 'conf', None)
        force_single_line = getattr(self, 'force_single_line', False)
        if not force_single_line and isinstance(conf, dict):
            force_single_line = conf.get('force_single_line', False)
        return force_single_line

    def _set_y_band_from_spread(self, y_coords):
        """Set adaptive_y_min/max + is_multiline from the contour y-spread."""
        y_min = min(y_coords)
        y_max = max(y_coords)
        scaled_params = self._get_scaled_params()
        y_margin = scaled_params['y_range_margin']
        self.is_multiline = (y_max - y_min) > scaled_params['multiline_threshold']
        if self.is_multiline:
            # Permissive band preserving every line.
            self.adaptive_y_min = max(0, y_min - y_margin)
            self.adaptive_y_max = y_max + y_margin
        else:
            # Single-line: full image height (permissive for cropped images).
            self.adaptive_y_min = 0
            self.adaptive_y_max = self.img_arr.shape[0]

    def _analyze_document_type(self, contours):
        """Analyze document layout and set adaptive filtering ranges (adaptive_y_min/
        max + is_multiline)."""
        if self._is_force_single_line():
            self.adaptive_y_min = 0
            self.adaptive_y_max = self.img_arr.shape[0]
            self.is_multiline = False
            return

        y_coords = [cv.boundingRect(contour)[1] for contour in contours]
        if y_coords:
            self._set_y_band_from_spread(y_coords)
        else:
            # Fallback: single-line with scale-adaptive ranges.
            scaled_params = self._get_scaled_params()
            self.adaptive_y_min = max(1, int(5 * scaled_params['scale_factor']))
            self.adaptive_y_max = max(self.adaptive_y_min + 10, int(35 * scaled_params['scale_factor']))
            self.is_multiline = False

    def update_shapes(self):
        # _contours() always returns (contours, hierarchy) (version-agnostic unpack
        # is done inside it). Use a length-based check like PageElements.__init__ —
        # the old platform.system() branch raised ValueError on non-Linux.
        contour_result = self._contours()
        if len(contour_result) == 2:
            self.contours, self.hierarchy = contour_result
        else:
            _, self.contours, self.hierarchy = contour_result

        self.boxes = self._boxes()
        self._set_shape_measurements()
        self.indices = [i for i, b in enumerate(self.get_boxes()) if (
               max(b[2], b[3]) <= 6 * self.char_mean )]

    def _draw_new_page(self):
        self.page_array = np.ones_like(self.img_arr)

        self.tall = set([i for i in self.get_indices() if
                         self.get_boxes()[i][3] > 3*self.char_mean])

        cv.drawContours(self.page_array, [self.contours[i] for i in
                        range(len(self.contours)) if
                        self.get_boxes()[i][2] > self.smlmean + 3*self.smstd],
                        -1,0, thickness = -1)
        import Image
        Image.fromarray(self.page_array*255).show()

    def _collect_char_contours(self, root_ind):
        """Root contour plus its first child and that child's sibling chain."""
        char_contours = [root_ind]
        root = self.hierarchy[0][root_ind]
        if root[2] >= 0:
            char_contours.append(root[2])  # root's first child
            child_hier = self.hierarchy[0][root[2]]
            has_sibling = child_hier[0] >= 0
            while has_sibling:
                ind = child_hier[0]  # sibling's index
                char_contours.append(ind)
                child_hier = self.hierarchy[0][ind]
                if child_hier[0] < 0:
                    has_sibling = False
        return char_contours

    def _map_contours(self, char_contours):
        """Resolve contour indices to contour arrays, honoring the syllable-merge
        index remap (skipping merged subscripts) when one is present."""
        if hasattr(self, 'contour_index_mapping') and self.contour_index_mapping is not None:
            mapped_contours = []
            for j in char_contours:
                if j in self.contour_index_mapping:
                    new_index = self.contour_index_mapping[j]
                    if new_index is not None and new_index < len(self.contours):
                        mapped_contours.append(self.contours[new_index])
            return mapped_contours
        return [self.contours[j] for j in char_contours]

    def draw_contour_and_children(self, root_ind, char_arr=None, offset=()):
        char_contours = self._collect_char_contours(root_ind)
        if not hasattr(char_arr, 'dtype'):
            x, y, w, h = self.get_boxes()[root_ind]
            char_arr = np.ones((h, w), dtype=np.uint8)
            offset = (-x, -y)
        cv.drawContours(char_arr, self._map_contours(char_contours), -1, 0, thickness=-1, offset=offset)
        return char_arr
