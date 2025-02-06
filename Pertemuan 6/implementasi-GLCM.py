import cv2
from skimage.feature import graycomatrix, graycoprops

# Membaca gambar dalam grayscalep
image = cv2.imread('EEDAN.jpg', 0)

# Menghitung GLCM
glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# Menghitung fitur tekstur dari GLCM
contrast = graycoprops(glcm, 'contrast')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]

print(f'Contrast: {contrast}; Energy: {energy}')