import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=2))
model.add(tf.keras.layers.Dense(1,activation='sigmoid',use_bias=True))

input = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
output = np.array([0.0, 1.0, 1.0, 1.0])

pred_1 = model(input).numpy()

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(input, output, epochs=5000)

print('Salida antes de entrenar:')
print(pred_1)

pred = model(input).numpy()
print('Salida despues de entrenar:')
print(pred)