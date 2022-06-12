import tensorflow as tf

### Hyperparameters
try:
    AUTOTUNE = tf.data.AUTOTUNE     
except:
    AUTOTUNE = tf.data.experimental.AUTOTUNE

shuffle_buffer = 5000
# The below two values are taken from https://www.tensorflow.org/datasets/catalog/stl10
labelled_train_images = 5000
unlabelled_images = 100000

temperature = 0.1
queue_size = 10000
contrastive_augmenter = {
    "brightness": 0.5,
    "name": "contrastive_augmenter",
    "scale": (0.2, 1.0),
}
classification_augmenter = {
    "brightness": 0.2,
    "name": "classification_augmenter",
    "scale": (0.5, 1.0),
}
input_shape = (96, 96, 3)
width = 128
num_epochs = 25
steps_per_epoch = 200