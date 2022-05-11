import cv2
import numpy as np
from typing import Tuple
'''
ImageEnhancement equalizes the histogram of the image
'''
def ImageEnhancement(normalized):
    enhanced = []
    for res in normalized:
        res = res.astype(np.uint8)
        im = cv2.equalizeHist(res)
        enhanced.append(im)
    return np.stack(enhanced).squeeze()

'''
Converting Cartesian coordinates to Polar coordinates
Returns a list of 64 x 512 normalized images
'''

def normalize(boundary,centers, iris_radius):
    target = [img for img in boundary]
    normalized = []
    cent = 0
    for img in target:
        # Loading pupil centers and radius of inner circles
        center_x = centers[cent][0]
        center_y = centers[cent][1]
        radius_pupil=int(centers[cent][2])
        
        nsamples = 360
        samples = np.linspace(0,2*np.pi, nsamples)[:-1]
        polar = np.zeros((iris_radius, nsamples))
        for r in range(iris_radius):
            for theta in samples:
                # Get x and y for values on inner boundary
                x = (r+radius_pupil)*np.cos(theta)+center_x
                y = (r+radius_pupil)*np.sin(theta)+center_y
                x=int(x)
                y=int(y)
                try:
                # Convert coordinates
                    polar[r][int((theta*nsamples)/(2*np.pi))] = img[y][x]
                except IndexError: # Ignores values which lie out of bounds
                    pass
                continue
        res = cv2.resize(polar,(512,64))
        normalized.append(res)
        cent+=1
    return normalized 

def normalizeNonconcentric(image : np.ndarray, iris_center: Tuple[int,int], iris_radius: int, pupil_center: Tuple[int,int], pupil_radius: int, nsamples = 360, relative_rotation = 0.0):
    polar = np.zeros((iris_radius, nsamples))
    samples = np.linspace(0, 2 * np.pi, nsamples + 1)[:-1] + relative_rotation
    for r_idx, r in enumerate(np.linspace(0, 1, iris_radius)):
        for theta_idx, theta in enumerate(samples):
            x_iris = iris_center[0] + iris_radius * np.cos(theta)
            y_iris = iris_center[1] + iris_radius * np.sin(theta)

            x_pupil = pupil_center[0] + pupil_radius * np.cos(theta)
            y_pupil = pupil_center[1] + pupil_radius * np.sin(theta)

            x = (1 - r) * x_pupil + r * x_iris
            y = (1 - r) * y_pupil + r * y_iris
            x, y = int(x), int(y)
            try:
                polar[r_idx][theta_idx] = image[y][x]
            except IndexError:
                pass
            continue
    res = cv2.resize(polar, (512, 64))
    return res