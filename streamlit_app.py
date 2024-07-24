import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import soundfile as sf

# Загрузка обученной модели
model_dir = 'artifacts/model/data/model.keras'
model = tf.keras.models.load_model(model_dir)

# Определение функций для обработки аудиофайлов и извлечения признаков
def extract_features(audio, sample_rate, frame_length, feature_type, num_mfcc_features):
    samples_per_frame = int(sample_rate * frame_length)
    total_frames = int(len(audio) / samples_per_frame)
    features = []

    for i in range(total_frames):
        start_idx = i * samples_per_frame
        end_idx = start_idx + samples_per_frame
        frame = audio[start_idx:end_idx]

        if feature_type == 'mfcc':
            feature = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=num_mfcc_features).T
        elif feature_type == 'raw':
            feature = frame.reshape(-1, 1)  # Переформатирование для согласованности с (длина последовательности, количество признаков)
        features.append(feature)

    features = np.array(features)
    # Если используется сырой аудио, дополняем последовательности до одинаковой длины
    if feature_type == 'raw':
        max_length = max(len(f) for f in features)
        features = np.array([np.pad(f, ((0, max_length - len(f)), (0, 0)), 'constant') for f in features])
        features = np.expand_dims(features, -1)  # Добавляем измерение для количества признаков

    return features

# Приложение Streamlit
st.title('Приложение для предсказания на основе аудио')

# Загрузка аудиофайла
uploaded_file = st.file_uploader('Загрузите аудиофайл', type=['wav'])

if uploaded_file is not None:
    # Загрузка аудиофайла
    audio, sample_rate = librosa.load(uploaded_file, sr=44100)
    st.audio(uploaded_file, format='audio/wav')

    # Извлечение признаков
    features = extract_features(audio, sample_rate, frame_length=0.05, feature_type='raw', num_mfcc_features=13)

    # Выполнение предсказаний
    predictions = model.predict(features).flatten()

    # Отображение предсказаний
    st.write('Предсказанные значения:')
    st.write(predictions)
