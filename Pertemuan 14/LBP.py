import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Baca citra dalam grayscale
image = cv2.imread('EEDAN.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan Local Binary Pattern (LBP)
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

# Tampilkan hasil LBP
cv2.imshow('Local Binary Pattern', lbp.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()