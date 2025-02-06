import cv2
import numpy as np

# Baca citra 
image_gray = cv2.imread('EEDAN.jpg', cv2.IMREAD_GRAYSCALE)


# Konversi citra ke grayscale 
image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

# Terapkan thresholding
ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Penghapusan noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Tentukan area latar belakang
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Tentukan area objek
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Temukan area perbatasan
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling
ret, markers = cv2.connectedComponents(sure_fg)

# Tambahkan 1 ke semua marker sehingga background akan menjadi 1, bukan 0
markers = markers + 1

# Tandai area perbatasan dengan 0
markers[unknown == 255] = 0

# Konversi markers 
markers = markers.astype(np.int32)

# Terapkan Watershed
markers = cv2.watershed(image, markers)

# Tandai batas dengan warna merah
image[markers == -1] = [255, 0, 0]

# Tampilkan hasil segmentasi
cv2.imshow('Segmented Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()