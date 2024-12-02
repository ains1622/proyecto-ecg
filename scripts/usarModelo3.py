import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

# Ruta al modelo guardado y al archivo CSV con tus datos de ECG
ruta_modelo = 'modelo_ecg.h5'
ruta_csv = 'sensor_data.csv'  # Cambia esto por la ruta de tu archivo CSV

# Longitud fija para cada latido (asegúrate de que tus datos coincidan con esto)
tam_latido = 100

# Cargar el modelo guardado
modelo_cargado = load_model(ruta_modelo)
print("Modelo cargado con éxito.")

# Cargar datos desde el archivo CSV
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


archivo_csv = 'data/sensor_data.csv'  # Reemplaza con la ruta a tu archivo CSV
df = pd.read_csv(archivo_csv)

# Asegúrate de que la columna 'value' existe en el archivo CSV
if 'value' not in df.columns:
    raise ValueError("La columna 'value' no existe en el archivo CSV.")

valores = df['value'].values

# Determinar el número de elementos por fila (por ejemplo, si quieres 100 elementos por fila)
tam_fila = 100  # Ajusta esto al número que desees
n_filas = len(valores) // tam_fila  # Número de filas completas que puedes crear

# Cortamos los datos para que sea divisible entre tam_fila (si es necesario)
valores_cortados = valores[:n_filas * tam_fila]

# Reformateamos los datos en una matriz 2D (cada fila tiene 'tam_fila' elementos)
datos_matrix = np.reshape(valores_cortados, (n_filas, tam_fila))


# Cargar y preprocesar los datos
datos = datos_matrix
datos_escalados, scaler = escalar_datos_csv(datos)

# Asegurarse de que los datos tienen la forma correcta para el modelo (n_samples, tam_latido, 1)
datos_escalados = datos_escalados.reshape(-1, tam_latido, 1)

# Hacer las predicciones
predicciones = modelo_cargado.predict(datos_escalados)

# Decodificar las predicciones
# Si las clases son 'Normal' (0) y 'Anomalía' (1), puedes obtener la clase predicha con argmax
clases_predichas = np.argmax(predicciones, axis=1)

# Si tienes más de 2 clases, puedes ajustar el codificador según tu entrenamiento
etiquetas_predichas = ['Normal' if clase == 0 else 'Anomalía' for clase in clases_predichas]

# Mostrar las predicciones
for i, etiqueta in enumerate(etiquetas_predichas):
    print(f"Latido {i+1}: {etiqueta}")

# Si deseas guardar las predicciones en un archivo CSV
df_predicciones = pd.DataFrame({
    'Latido': np.arange(1, len(etiquetas_predichas) + 1),
    'Predicción': etiquetas_predichas
})
df_predicciones.to_csv('predicciones_ecg.csv', index=False)
print("Predicciones guardadas en 'predicciones_ecg.csv'")