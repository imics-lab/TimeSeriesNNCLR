from tensorflow import keras
from tensorflow.keras import layers
from config import *

# def encoder():
#     return keras.Sequential(
#         [
#             layers.Input(shape=input_shape),
#             layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
#             layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
#             layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
#             layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
#             layers.Flatten(),
#             layers.Dense(width, activation="relu"),
#         ],
#         name="encoder",
#     )

def encoder():
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=100, kernel_size=k_size, activation='relu'),
			layers.Conv1D(filters=100, kernel_size=k_size, activation='relu'),
			layers.Dropout(0.5),
			layers.MaxPooling1D(pool_size=2),
			layers.Flatten(),
			layers.Dense(width, activation='relu'),
        ],
        name="encoder",
    )