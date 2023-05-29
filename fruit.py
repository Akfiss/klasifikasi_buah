import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Fungsi untuk mengekstraksi fitur berdasarkan statistika citra
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    return hist

# Membaca dataset citra buah
def load_dataset():
    images = []
    labels = []
    
    # Ubah path direktori dataset sesuai dengan dataset Anda
    dataset_dir = 'D:/VS Code/fruit_classification/'
    for i in range(1, 6):  # Terdapat 5 tingkatan kematangan buah
        for j in range(1, 21):  # Terdapat 100 citra untuk setiap tingkatan kematangan buah
            image_path = dataset_dir + '/tingkat' + str(i) + '/buah' + str(j) + '.jpg'
            image = cv2.imread(image_path)
            feature = extract_features(image)
            images.append(feature)
            labels.append(i)
    
    return images, labels

# Membagi dataset menjadi data latih dan data uji
def split_dataset(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Melatih model klasifikasi menggunakan Support Vector Machine (SVM)
def train_model(X_train, y_train):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return svm

# Mengevaluasi model menggunakan data uji
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, confusion, y_pred

# Menampilkan histogram metrik evaluasi dan hasil tingkat kematangan buah
def plot_histogram(accuracy, precision, recall, y_pred):
    labels = ['Akurasi', 'Presisi', 'Recall']
    values = [accuracy, precision, recall]

    plt.subplot(1, 2, 1)
    plt.bar(labels, values)
    plt.ylabel('Nilai')
    plt.title('Metrik Evaluasi')

    plt.subplot(1, 2, 2)
    plt.hist(y_pred, bins=np.arange(1, 7)-0.5, edgecolor='black', alpha=0.8)
    plt.xticks(range(1, 6))
    plt.xlabel('Tingkat Kematangan')
    plt.ylabel('Frekuensi')
    plt.title('Tingkat Kematangan Buah (Hasil Prediksi)')

    plt.tight_layout()
    plt.show()

# Memuat dataset
images, labels = load_dataset()

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = split_dataset(images, labels)

# Melatih model klasifikasi
model = train_model(X_train, y_train)

# Evaluasi model dan memperoleh hasil
accuracy, precision, recall, confusion, y_pred = evaluate_model(model, X_test, y_test)

# Menampilkan histogram metrik evaluasi dan hasil tingkat kematangan buah
plot_histogram(accuracy, precision, recall, y_pred)
