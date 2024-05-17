from tensorflow import keras
from tensorflow.keras import layers

# Encoder architecture. This can be changed to any other architecture.
def encoder(input_shape, output_width):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=100, kernel_size=7, activation='relu'),
			layers.Conv1D(filters=100, kernel_size=7, activation='relu'),
			layers.Dropout(0.5),
			layers.MaxPooling1D(pool_size=2),
			layers.Flatten(),
			layers.Dense(output_width, activation='relu'),
        ],
        name="encoder",
    )