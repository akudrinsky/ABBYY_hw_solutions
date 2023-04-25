import cv2 as cv
import numpy as np
from typing import Iterable, Optional, Tuple


class QrDetector():
    def __init__(self,
                 blur_kernel_size=7,
                 gray_grad_threshold=170,
                 rectangle_kernel_size=9,
                 grey_crop_kernel_size=19,
                 n_candidate_contours=5,
                 squareness_factor=0.2,
                 area_threshold=25,
                 rectangular_area_factor_threshold=0.7,
                 pattern_color=(255, 0, 0)
                 ):
        self.blur_kernel_size = blur_kernel_size
        self.gray_grad_threshold = gray_grad_threshold
        self.rectangle_kernel_size = rectangle_kernel_size
        self.grey_crop_kernel_size = grey_crop_kernel_size
        self.n_candidate_contours = n_candidate_contours
        self.squareness_factor = squareness_factor
        self.area_threshold = area_threshold
        self.rectangular_area_factor_threshold = rectangular_area_factor_threshold
        self.pattern_color = pattern_color

    def __call__(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = self._get_grey_image(image)
        contours = self._find_candidate_contours(gray)

        at_least_one_qr = False
        for contour in contours:
            (x_min, y_min), (x_max, y_max) = self._get_contour_bounds(contour)
            patterns = self._qr_findpatterns(image[y_min:y_max, x_min:x_max])
            if patterns is None:
                continue
            patterns = patterns + [x_min, y_min]
            image = self.draw_patterns(image, patterns)
            at_least_one_qr = True

        return image if at_least_one_qr else None

    def draw_patterns(self, image: np.ndarray, patterns: np.ndarray) -> np.ndarray:
        image = image.copy()
        x, y, w, h = cv.boundingRect(patterns.reshape(-1, 2))
        points = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])]
        cv.drawContours(image, points, -1, self.pattern_color, 3, cv.LINE_AA)
        cv.drawContours(image, patterns, -1, self.pattern_color, 3, cv.LINE_AA)
        return image

    def _qr_findpatterns(self, image) -> Optional[np.ndarray]:
        """Find qr code findpatterns based on the following criteria: it is a three-nested square contour. """
        thresh = self._get_thresh_crop(image)
        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None

        contours, boxes, hierarchy, orig2new_mapping = self._filter_nonsquare_contours(
            contours, hierarchy[0])

        findpattern_indexes = []
        for k in range(len(hierarchy)):
            child_id = hierarchy[k][2]
            if not self._check_valid_square(child_id, orig2new_mapping):
                continue

            grandchild_id = hierarchy[orig2new_mapping[child_id]][2]
            if not self._check_valid_square(grandchild_id, orig2new_mapping):
                continue

            findpattern_indexes.append(k)

        if len(findpattern_indexes) != 3: # Valid qr code has only 3 find patterns
            return None
        return np.array([boxes[i] for i in findpattern_indexes])

    def _check_valid_square(self, index, valid_index_set: Iterable):
        """ Need valid index by cv documentation and to be in set of valid indices. """
        return index != -1 and index in valid_index_set

    def _filter_nonsquare_contours(self, contours,  hierarchy):
        new_contours = []
        boxs = []
        new_hierarchy = []
        orig2new_mapping = {}

        for i, contour in enumerate(contours):
            rect = cv.minAreaRect(contour)
            center_point, (width, hight), angle = rect
            box = cv.boxPoints(rect).astype(int)

            # Nearly square shape
            if abs(hight - width) > self.squareness_factor * max(hight, width):
                continue
            # Big enough
            if cv.contourArea(box) < self.area_threshold:
                continue
            # Сircumscribed recrangle is similar enough
            if cv.contourArea(contour) < self.rectangular_area_factor_threshold * cv.contourArea(box):
                continue

            orig2new_mapping[i] = len(new_contours)
            new_contours.append(contour)
            boxs.append(box)
            new_hierarchy.append(hierarchy[i])

        return new_contours, boxs, new_hierarchy, orig2new_mapping

    def _get_thresh_crop(self, image: np.ndarray) -> np.ndarray:
        grey = self._get_grey_image(image)
        thresh = cv.adaptiveThreshold(
            grey, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, self.grey_crop_kernel_size, 1)
        thresh = cv.medianBlur(thresh, 5, 5)
        return thresh

    def _get_grey_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype('uint8')
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return image

    def _find_candidate_contours(self, image: np.ndarray) -> np.ndarray:
        gradient_x = cv.Sobel(image, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
        gradient_y = cv.Sobel(image, ddepth=cv.CV_32F, dx=0, dy=1, ksize=-1)

        gradient = gradient_x - gradient_y
        gradient = cv.convertScaleAbs(gradient)

        blurred = cv.blur(
            gradient, (self.blur_kernel_size, self.blur_kernel_size))
        _, thresh = cv.threshold(
            blurred, self.gray_grad_threshold, 255, cv.THRESH_BINARY)

        kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (self.rectangle_kernel_size, self.rectangle_kernel_size))
        closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        closed = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)

        contours, _ = cv.findContours(
            image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[
            :self.n_candidate_contours]

        return contours

    def _get_contour_bounds(self, contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect).astype(int)
        return np.min(box, axis=0),  np.max(box, axis=0)
