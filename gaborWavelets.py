import numpy as np 
import numpy as np
from scipy import ndimage as ndi
from skimage.util import view_as_blocks

def defined_gabor_kernel(frequency, sigma_x = None, sigma_y = None, n_stds = 3, offset = 0, theta = 0):
    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(theta)),
                        np.abs(n_stds * sigma_y * np.sin(theta)), 1))
    y0 = np.ceil(max(np.abs(n_stds * sigma_y * np.cos(theta)),
                        np.abs(n_stds * sigma_x * np.sin(theta)), 1))
    y, x = np.mgrid[-y0:y0 + 1, -x0:x0 + 1]

    g = np.zeros(y.shape, dtype=np.complex)
    g[:] = np.exp(-0.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y

    g *= (np.cos(2 * np.pi * frequency * ((x ** 2 + y ** 2) ** 0.5)) + 
        1j * np.sin(2 * np.pi * frequency * ((x ** 2 + y ** 2) ** 0.5)))
    #g *= np.exp(1j * 2 * np.pi * frequency * ((x ** 2 + y ** 2) ** 0.5))
    return g

def defined_gabor(img, frequency, sigma_x, sigma_y):
    g = defined_gabor_kernel(frequency, sigma_x, sigma_y)
    filtered_real = ndi.convolve(img, np.real(g), mode='wrap', cval=0)
    filtered_imag = ndi.convolve(img, np.imag(g), mode='wrap', cval=0)

    return filtered_real, filtered_imag

def getBitBlocks(filtered, size = 8):
    blocks = view_as_blocks(filtered, block_shape = (size,size)).reshape([-1,size**2])
    blocks = np.mean(blocks, axis = -1) - np.mean(filtered)
    return np.maximum(np.sign(blocks), 0)