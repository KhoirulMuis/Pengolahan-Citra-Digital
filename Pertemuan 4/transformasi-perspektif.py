import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('EEDAN.jpg')

# Mendefinisikan empat titik sudut citra asli
points1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

# Mendefonisikan empat titik sudut baru
points2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

# Mendapatkan matriks transformasi perspektif
M_perspective = cv2.getPerspectiveTransform(points1, points2)

# Melakukan transformasi perspektif
perspective_tranformed_image = cv2.warpPerspective(image, M_perspective, (300, 300))

# Menampilkan hasil
cv2.imshow('Perspective Transformed Image', perspective_tranformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()