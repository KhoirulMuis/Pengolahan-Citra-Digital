import cv2
import numpy as np

# Baca citra
image = cv2.imread('EEDAN.jpg')
resize = cv2.resize(image, (500, 500))
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

# Lakukan thresholding untuk menghilangkan noise
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological operation untuk memisahkan background dan foreground
kernel = np.ones((3, 3), np.uint8)
sure_bg = cv2.dilate(thresh, kernel, iterations=3)

# Distance transform untuk mendapatkan foreground
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Watershed
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Tandai komponen
ret, markers = cv2.connectedComponents(sure_fg)

# Tambahkan 1 pada marker sehingga background menjadi 1
markers = markers + 1
markers[unknown == 255] = 0

# Terapkan Watershed
markers = cv2.watershed(resize, markers)
resize[markers == -1] = [255, 0, 0]

# Tampilkan hasil
cv2.imshow('Watershed Segmentation', resize)
cv2.waitKey(0)
cv2.destroyAllWindows()