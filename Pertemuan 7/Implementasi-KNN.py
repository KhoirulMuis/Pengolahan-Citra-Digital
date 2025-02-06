import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Membaca dataset
data = pd.read_csv('simple_cars_dataset (1).csv')

# Encoding label menjadi numerik
Label_encoder = LabelEncoder()
data['label'] = Label_encoder.fit_transform(data['label'])

# Memisahkan fitur (x) dan label (y)
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Memisahkan data menjadi data latih dan uji
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Membuat dan melatih model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Prediksi pada data uji
y_pred = knn.predict(x_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN Accuracy: {accuracy * 100:.2f}%')