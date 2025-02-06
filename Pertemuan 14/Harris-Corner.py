import cv2
import numpy as np

# Baca citra dalam grayscale
image = cv2.imread('EEDAN.jpg', cv2.IMREAD_GRAYSCALE)

# Konversi gambar grayscale 
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Terapkan Harris Corner Detector
gray = np.float32(image)
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilasi untuk menonjolkan sudut
corners = cv2.dilate(corners, None)

# Tingkatkan sudut yang terdeteksi
image_color[corners > 0.01 * corners.max()] = [0, 0, 255]

# Tampilkan hasil deteksi sudut
cv2.imshow('Harris Corner Detection', image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()