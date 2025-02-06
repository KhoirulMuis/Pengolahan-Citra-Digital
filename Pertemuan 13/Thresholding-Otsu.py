import cv2

# Baca citra dalam grayscale
image = cv2.imread('EEDAN.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan Otsu's thresholding
ret, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Tampilkan hasil segmentasi
cv2.imshow('Otsu Thresholding', otsu_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()