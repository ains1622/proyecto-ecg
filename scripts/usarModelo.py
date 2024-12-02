import tensorflow as tf
import numpy as np

# Cargar el modelo entrenado
model = tf.keras.models.load_model('/autoencoder.weights.h5')

# Ejemplo de datos de entrada
example_data = np.random.rand(1000, 1) # Aquí hay que poner los datos para detectar anomalías


predictions = model.predict(example_data)
print("Predicciones:", predictions)
