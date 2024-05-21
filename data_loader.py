import sys
import os
import tensorflow as tf
# from config import *

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000


def load_e4_dataset():
    # Get project root directory path
    my_path = os.path.abspath(os.path.dirname(__file__))
    full_filename = os.path.join(my_path, 'load_data_time_series', 'HAR', 'e4_wristband_Nov2019')
    # Add the path to your pythonpath
    sys.path.append(full_filename)
    from e4_load_dataset import e4_load_dataset
    
    x_train, y_train, x_validate, y_validate, x_test, y_test = e4_load_dataset(incl_val_group = True, incl_xyz_accel= True, one_hot_encode = False)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_validate shape:", x_validate.shape)
    print("y_validate shape:", y_validate.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Create tf.data.Dataset objects
    unlabeled_train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    unlabeled_train_dataset = unlabeled_train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    unlabeled_train_dataset = unlabeled_train_dataset.prefetch(AUTOTUNE)

    labeled_train_dataset = tf.data.Dataset.from_tensor_slices((x_validate, y_validate))
    labeled_train_dataset = labeled_train_dataset.batch(BATCH_SIZE)
    labeled_train_dataset = labeled_train_dataset.prefetch(AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=AUTOTUNE)

    return BATCH_SIZE, train_dataset, unlabeled_train_dataset, labeled_train_dataset, test_dataset, x_train, y_train, x_validate, y_validate, x_test, y_test


def load_unimib_dataset():
    # Get project root directory path
    my_path = os.path.abspath(os.path.dirname(__file__))
    full_filename = os.path.join(my_path, 'load_data_time_series', 'HAR', 'UniMiB_SHAR')
    # Add the path to your pythonpath
    sys.path.append(full_filename)
    from unimib_shar_adl_load_dataset import unimib_load_dataset
    
    x_train, y_train, x_validate, y_validate, x_test, y_test = unimib_load_dataset(incl_val_group = True, incl_xyz_accel= True, one_hot_encode = False)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_validate shape:", x_validate.shape)
    print("y_validate shape:", y_validate.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Create tf.data.Dataset objects
    unlabeled_train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    unlabeled_train_dataset = unlabeled_train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    unlabeled_train_dataset = unlabeled_train_dataset.prefetch(AUTOTUNE)

    labeled_train_dataset = tf.data.Dataset.from_tensor_slices((x_validate, y_validate))
    labeled_train_dataset = labeled_train_dataset.batch(BATCH_SIZE)
    labeled_train_dataset = labeled_train_dataset.prefetch(AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=AUTOTUNE)

    return BATCH_SIZE, train_dataset, unlabeled_train_dataset, labeled_train_dataset, test_dataset, x_train, y_train, x_validate, y_validate, x_test, y_test