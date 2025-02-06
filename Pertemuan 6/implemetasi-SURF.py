import cv2

# Membaca gambar
image = cv2.imread('EEDAN.jpg')

# Inisialisasi objek SURF
orb = cv2.ORB_create()

# Mendeteksi keypoints dan komputasi deskriptor
keypoints, descriptors = orb.detectAndCompute(image, None)

# Menggambar keypoints di citra
surf_image = cv2.drawKeypoints(image, keypoints, None)

# Menampilkan Hasil
cv2.imshow('SURF Features', surf_image)
cv2.waitKey(0)
cv2.destroyWindow()