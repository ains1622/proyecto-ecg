import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


import pandas as pd
import numpy as np

# Leer el archivo CSV
archivo_csv = 'data/sensor_data.csv'  # Reemplaza con la ruta a tu archivo CSV
df = pd.read_csv(archivo_csv)

# Asegúrate de que la columna 'value' existe en el archivo CSV
if 'value' not in df.columns:
    raise ValueError("La columna 'value' no existe en el archivo CSV.")

# Extraer la columna 'value'
valores = df['value'].values

# Determinar el número de elementos por fila (por ejemplo, si quieres 100 elementos por fila)
tam_fila = 100  # Ajusta esto al número que desees
n_filas = len(valores) // tam_fila  # Número de filas completas que puedes crear

# Cortamos los datos para que sea divisible entre tam_fila (si es necesario)
valores_cortados = valores[:n_filas * tam_fila]

# Reformateamos los datos en una matriz 2D (cada fila tiene 'tam_fila' elementos)
datos_matrix = np.reshape(valores_cortados, (n_filas, tam_fila))

# Ahora 'datos_matrix' tiene la forma deseada
print(datos_matrix)


# Ruta al modelo guardado y al archivo CSV
ruta_modelo = 'autoencoder_ecg.h5'
ruta_csv = 'sensor_data.csv'

import tensorflow as tf

# Registrar el nombre de la métrica 'mse' manualmente
tf.keras.losses.MeanSquaredError = tf.keras.losses.MeanSquaredError

# Cargar el modelo
autoencoder = load_model(ruta_modelo, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
print("Modelo cargado con éxito.")

def cargar_csv(ruta_csv):
    datos = pd.read_csv(ruta_csv, header=None)  # Asegúrate de que no haya encabezados si son puramente datos
    return datos.values  # Devuelve como array de NumPy

# Preprocesar los datos (escalado entre 0 y 1)
def escalar_datos_csv(datos, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        datos_escalados = scaler.fit_transform(datos)
    else:
        datos_escalados = scaler.transform(datos)
    return datos_escalados, scaler

# Cargar y preprocesar los datos
#datos = cargar_csv(ruta_csv)
datos = datos_matrix
datos_escalados, scaler = escalar_datos_csv(datos)

# Predicción con el modelo cargado
reconstrucciones = autoencoder.predict(datos_escalados)

# Calcular el error de reconstrucción
errores = np.mean(np.square(datos_escalados - reconstrucciones), axis=1)

# Establecer un umbral para clasificar anomalías
umbral = np.percentile(errores, 95)  # Ejemplo: el 95% de los datos son normales
anomalías = errores > umbral

# Resultados
for i, es_anomalía in enumerate(anomalías):
    estado = "Anómalo" if es_anomalía else "Normal"
    print(f"Señal {i+1}: {estado} (Error: {errores[i]:.4f})")

# Visualizar ejemplos de señales
plt.figure(figsize=(12, 6))
for i in range(min(5, len(datos))):  # Mostrar hasta 5 ejemplos
    plt.subplot(2, 5, i + 1)
    plt.plot(datos_escalados[i], label='Original')
    plt.plot(reconstrucciones[i], label='Reconstruida')
    plt.title(f'Señal {i+1}')
    plt.legend()
plt.tight_layout()
plt.show()