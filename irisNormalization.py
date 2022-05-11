import cv2
import numpy as np

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