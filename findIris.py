import cv2
import numpy as np
import itertools
from typing import Tuple
import math

# Code from https://github.com/banderlog/daugman/blob/master/daugman.py
# Slightly modified.

'''
Function that will crop the image array and convert it to a grayscale iamge
'''
def preprocess_image(imageArray):
    gray_img = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
    h,w = gray_img.shape

    gray_img = gray_img[:,(w - h)//2:(w + h)//2]
    return gray_img

'''
Function that will crop the image after finding the outer iris boundary, 
to facilitate to find the pupil
'''
def preprocess_pupil(gray_img, iris_center, iris_rad):
    pupil_crop = gray_img[(iris_center[1] - iris_rad):(iris_center[1] + iris_rad), 
    (iris_center[0] - iris_rad):(iris_center[0] + iris_rad)]
    return pupil_crop

'''
Daugman's algorithm for finding circles
'''
def daugman(gray_img: np.ndarray, center: Tuple[int, int], start_r: int, end_r: int, step: int = 1) -> Tuple[float, int]:
    x, y = center
    intensities = []
    mask = np.zeros_like(gray_img)

    # For every radius in range
    radii = list(range(start_r, end_r, step))
    for r in radii:
        # Draw circle on mask
        cv2.circle(mask, center, r, 255, 1)
        # Get pixel from original image
        diff = gray_img & mask
        # Normalization
        intensities.append(np.add.reduce(diff[diff > 0]) / (2 * math.pi * r))
        # Refresh mask
        mask.fill(0)

    # Calculate delta of radius intensitiveness
    intensities_np = np.array(intensities, dtype=np.float32)

    # Circles intensity differences
    intensities_np = intensities_np[:-1] - intensities_np[1:]
    intensities_np = abs(cv2.GaussianBlur(intensities_np, (1, 5), 0))
    # Get maximum value
    idx = np.argmax(intensities_np)

    # Return intensity value, radius
    return intensities_np[idx], radii[idx]


def find_iris(gray: np.ndarray, *,
              daugman_start: int, daugman_end: int,
              daugman_step: int = 1, points_step: int = 1,) -> Tuple[Tuple[int, int], int]:

    h, w = gray.shape
    if h != w:
        print('Your image is not a square!')

    # Reduce step for better accuracy
    # We will look only on dots within central 1/3 of image
    single_axis_range = range(int(h / 3), h - int(h / 3), points_step)
    all_points = itertools.product(single_axis_range, single_axis_range)

    intensity_values = []
    coords = []  # List[Tuple[Tuple(int, int), int]]

    for point in all_points:
        val, r = daugman(gray, point, daugman_start, daugman_end, daugman_step)
        intensity_values.append(val)
        coords.append((point, r))

    # return the radius with biggest intensiveness delta on image
    # ((xc, yc), radius)
    best_idx = intensity_values.index(max(intensity_values))
    return coords[best_idx]

def find_iris_adjusted(gray: np.ndarray, *,
              daugman_start: int, daugman_end: int,
              daugman_step: int = 1, points_step: int = 1, scan_ratio = 0.85) -> Tuple[Tuple[int, int], int]:
    h, w = gray.shape
    assert h == w, f"Your image must be square, h = {h}, w = {w}"

    # Reduce step for better accuracy
    # We will look only on dots within central 1/3 of image
    single_axis_range = range(int(h * (1 - scan_ratio)), h - int(h * (1 - scan_ratio)), points_step)
    all_points = itertools.product(single_axis_range, single_axis_range)

    intensity_values = []
    coords = []  # List[Tuple[Tuple(int, int), int]]

    for point in all_points:
        distance_to_edge = min(h - point[0], h - point[1], point[0], point[1])
        daug_end = min(daugman_end, distance_to_edge - 3)
        if daug_end > daugman_start+daugman_step:
            val, r = daugman(gray, point, daugman_start, daug_end, daugman_step)
            intensity_values.append(val)
            coords.append((point, r))

    # return the radius with biggest intensiveness delta on image
    # ((xc, yc), radius)
    best_idx = intensity_values.index(max(intensity_values))
    return coords[best_idx]
