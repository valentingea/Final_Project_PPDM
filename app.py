import streamlit as st
import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
import tensorflow as tf
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Fungsi untuk menghitung fitur GLCM
def calculate_glcm_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    properties = [
        "dissimilarity",
        "correlation",
        "homogeneity",
        "contrast",
        "ASM",
        "energy",
    ]
    glcm = graycomatrix(gray_image, distances, angles, symmetric=True, normed=True)
    features = []
    for prop in properties:
        prop_values = graycoprops(glcm, prop)
        features.extend(prop_values.flatten())
    return features


# Fungsi untuk memuat model dan melakukan prediksi
def predict_emotion(features):
    model = tf.keras.models.load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
    scaled_features = scaler.transform(features)
    scaled_features = np.sqrt(scaled_features)
    predictions = model.predict(scaled_features)
    return predictions


# Judul halaman web
st.title("Deteksi Emosi")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Baca gambar yang diunggah
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    # Resize gambar
    target_width = 48
    target_height = 48
    resized_image = cv2.resize(image, (target_width, target_height))
    # Hitung fitur GLCM
    features = calculate_glcm_features(resized_image)
    features = np.array(features)
    st.write(features)
    # Prediksi emosi
    predictions = predict_emotion([features])
    emotion = "Happy" if predictions[0] > 0.55 else "Sad"


    # # Tampilkan gambar dan hasil prediksi
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.write("Prediksi Emosi:", emotion)