import cv2
import numpy as np
from math import log10

def calcular_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * log10(max_pixel / np.sqrt(mse))
