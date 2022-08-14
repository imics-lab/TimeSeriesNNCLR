import tensorflow as tf

### Hyperparameters
try:
    AUTOTUNE = tf.data.AUTOTUNE     
except:
    AUTOTUNE = tf.data.experimental.AUTOTUNE

# The below two values are taken from https://www.tensorflow.org/datasets/catalog/stl10
labelled_train_instances = 329
unlabelled_instances = 380

temperature = 0.1
queue_size = 100
# contrastive_augmenter = {
#     "brightness": 0.5,
#     "name": "contrastive_augmenter",
#     "scale": (0.2, 1.0),
# }
# classification_augmenter = {
#     "brightness": 0.2,
#     "name": "classification_augmenter",
#     "scale": (0.5, 1.0),
# }

input_shape = (96, 4)
width = 64
num_epochs = 25
# steps_per_epoch = 200
BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 50
k_size = 16
n_classes = 6

t_names = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']