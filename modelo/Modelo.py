import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pywt import wavedec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv('data/sensor_data.csv')

# Descartamos la columna 'id' (suponiendo que es la primera columna)
ecg_signal = data['value'].to_numpy()

# Convertimos la columna 'timestamp' a formato de tiempo adecuado
data['timestamp'] = pd.to_datetime(data['timestamp'])
time = (data['timestamp'] - data['timestamp'].iloc[0]).dt.total_seconds().to_numpy()  # Tiempo en segundos desde el inicio


# Filtro pasabajas para suavizar la señal
filtro_pasabajas = butter(4, 0.2, btype="low", fs=1000)  # Orden 4, frecuencia de corte 0.2 Hz
ecg_filtrada = filtfilt(filtro_pasabajas[0], filtro_pasabajas[1], ecg_signal)

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

ecg_normalizada = normalize(ecg_filtrada)

# Transformada wavelet discreta (Haar)
coeffs = wavedec(ecg_normalizada, wavelet='haar', level=4)
approximation = coeffs[0]
details = np.concatenate(coeffs[1:])

# Visualización de wavelets
plt.figure(figsize=(10, 6))
plt.plot(approximation, label="Aproximación (Wavelet)", color="green")
plt.plot(details, label="Detalles (Wavelet)", color="purple")
plt.legend()
plt.title("Coeficientes Wavelet")
plt.grid()
plt.show()

# Reestructuramos la señal normalizada para entrenar el modelo (usamos la señal como entrada)
X_train = np.reshape(ecg_normalizada, (len(ecg_normalizada), 1))

# Definir el modelo autoencoder
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(X_train.shape[1], activation='linear')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train, X_train, epochs=50, batch_size=16)

# Paso 6: Predicción (reconstrucción de la señal)
reconstructed = model.predict(X_train)

# Paso 7: Calcular el error de reconstrucción
error = np.mean((X_train - reconstructed) ** 2, axis=1)

# Paso 8: Detectar anomalías (basado en el umbral de error)
threshold = np.mean(error) + 2 * np.std(error)
anomalies = np.where(error > threshold)[0]

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.plot(time, ecg_signal, label="ECG Original", color="red")
plt.plot(time, ecg_filtrada, label="ECG Filtrada", color="blue")
plt.plot(time, reconstructed.flatten(), label="ECG Reconstruida", color="green")
plt.scatter(time[anomalies], ecg_signal[anomalies], color='black', label='Anomalías', zorder=5)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.title('Detección de Anomalías en Señales ECG')
plt.grid(True)
plt.show()

# Guardar pesos del modelo
model.save_weights("autoencoder.weights.h5")
print("Pesos del modelo guardados como 'autoencoder_weights.h5'")
