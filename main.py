# pip install tfds-nightly

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

from config import *
from augmentations import  RandomResizedCrop, RandomBrightness, augmenter
from encoder import encoder
from nnclr import NNCLR 


import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


### Load the dataset
dataset_name = "stl10"

def prepare_dataset():
    unlabeled_batch_size = unlabelled_images // steps_per_epoch
    labeled_batch_size = labelled_train_images // steps_per_epoch
    batch_size = unlabeled_batch_size + labeled_batch_size

    unlabeled_train_dataset = (
        tfds.load(
            dataset_name, split="unlabelled", as_supervised=True, shuffle_files=True
        )
        .shuffle(buffer_size=shuffle_buffer)
        .batch(unlabeled_batch_size, drop_remainder=True)
    )
    labeled_train_dataset = (
        tfds.load(dataset_name, split="train", as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=shuffle_buffer)
        .batch(labeled_batch_size, drop_remainder=True)
    )
    test_dataset = (
        tfds.load(dataset_name, split="test", as_supervised=True)
        .batch(batch_size)
        .prefetch(buffer_size=AUTOTUNE)
    )
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=AUTOTUNE)

    return batch_size, train_dataset, labeled_train_dataset, test_dataset


batch_size, train_dataset, labeled_train_dataset, test_dataset = prepare_dataset()



### Pre-train NNCLR
print("Pre-train:")
model = NNCLR(temperature=temperature, queue_size=queue_size)
model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
)
pretrain_history = model.fit(
    train_dataset, epochs=num_epochs, validation_data=test_dataset
)


### Evaluate our model
print("Fine tune:")
finetuning_model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        augmenter(**classification_augmenter),
        model.encoder,
        layers.Dense(10),
    ],
    name="finetuning_model",
)
finetuning_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

finetuning_history = finetuning_model.fit(
    labeled_train_dataset, epochs=num_epochs, validation_data=test_dataset
)


