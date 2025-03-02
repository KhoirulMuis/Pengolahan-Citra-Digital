import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Membaca dataset citra
def load_data():
    # Ganti dengan jalur dataset yang sesuai
    images = [] # List untuk citra
    labels = [] # List untuk label
    for i in range(1, 11): # Misalnya, 10 citra
        image = cv2.imread(f'FOOOT.jpeg', cv2.IMREAD_GRAYSCALE)
        images.append(image.flatten()) # Ubah citra menjadi vektor 1D
        labels.append(i) # Misalnya, label sesuai dengan nomor citra
    return np.array(images), np.array(labels)

# Muat data dan bagi menjadi data pelatihan dan pengujian
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latihan model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Uji model
accuracy = knn.score(X_test, y_test)
print(f'Akurasi KNN: {accuracy * 100:2f}%')