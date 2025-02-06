import cv2
import numpy as np

# Membaca gambar
image = cv2.imread("EEDAN.jpg")

# Mendapatkan dimensi gambar (tinggi dan lebar)
h, w = image.shape[:2]

# Membuat matriks transformasi affine
M_affine = np.float32([[1, 0, 100], [0, 1, 50]])

# Melakukan transformasi affine
affine_transformed_image = cv2.warpAffine(image, M_affine, (w, h))

# Menyimpan atau menampilkan gambar hasil
cv2.imshow("Affine Transform", affine_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

