import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Membaca dataset dari CSV
data = pd.read_csv('simple_cars_dataset (1).csv')

# Pastikan nama kolom benar
print("Kolom dalam dataset:", data.columns)

# Mengonversi label menjadi numerik
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])  # Gunakan nama kolom yang benar

# Memisahkan fitur (x) dan label (y)
x = data.iloc[:, :-1].values  # Semua kolom kecuali kolom terakhir
y = data.iloc[:, -1].values  # Kolom terakhir

# Memisahkan data menjadi data latih dan uji
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Membuat dan melatih model SVM
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

# Prediksi pada data uji
y_pred = clf.predict(x_test)

# Menghitung akurasi 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
