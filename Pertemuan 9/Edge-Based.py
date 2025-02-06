import cv2

# Baca citra grayscale
image = cv2.imread('EEDAN.jpg', cv2.IMREAD_GRAYSCALE)

resize = cv2.resize(image, (500, 500))

# Deteksi tepi menggunakan Canny
edges = cv2.Canny(resize, 70, 120)

# Tampilkan hasil
cv2.imshow('Edges Detected', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()