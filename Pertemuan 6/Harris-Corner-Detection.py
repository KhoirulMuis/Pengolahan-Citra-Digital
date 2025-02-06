import cv2
import numpy as np

# Membaca gambar dalam grayscale
image = cv2.imread('EEDAN.jpg', 0)
if image is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Periksa kembali path file.")

# Konversi gambar ke tipe float32
gray = np.float32(image)

# Menerapkan Harris Corner Detector
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilasi untuk menonjolkan sudut yang terdeteksi
dst = cv2.dilate(dst, None)

# Buat mask boolean
mask = dst > 0.01 * dst.max()

# Ubah gambar grayscale ke format BGR
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Tandai sudut pada gambar berwarna
image_color[mask] = [0, 0, 200]

# Menampilkan hasil
cv2.imshow('Harris Corners', image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()