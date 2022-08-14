from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from config import *
from time_series_augmentations import *

# class RandomResizedCrop(layers.Layer):
#     def __init__(self, scale, ratio):
#         super(RandomResizedCrop, self).__init__()
#         self.scale = scale
#         self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

#     def call(self, images):
#         batch_size = tf.shape(images)[0]
#         height = tf.shape(images)[1]
#         width = tf.shape(images)[2]

#         random_scales = tf.random.uniform((batch_size,), self.scale[0], self.scale[1])
#         random_ratios = tf.exp(
#             tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
#         )

#         new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
#         new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
#         height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
#         width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

#         bounding_boxes = tf.stack(
#             [
#                 height_offsets,
#                 width_offsets,
#                 height_offsets + new_heights,
#                 width_offsets + new_widths,
#             ],
#             axis=1,
#         )
#         images = tf.image.crop_and_resize(
#             images, bounding_boxes, tf.range(batch_size), (height, width)
#         )
#         return images


# class RandomBrightness(layers.Layer):
#     def __init__(self, brightness):
#         super(RandomBrightness, self).__init__()
#         self.brightness = brightness

#     def blend(self, images_1, images_2, ratios):
#         return tf.clip_by_value(ratios * images_1 + (1.0 - ratios) * images_2, 0, 1)

#     def random_brightness(self, images):
#         # random interpolation/extrapolation between the image and darkness
#         return self.blend(
#             images,
#             0,
#             tf.random.uniform(
#                 (tf.shape(images)[0], 1, 1, 1), 1 - self.brightness, 1 + self.brightness
#             ),
#         )

#     def call(self, images):
#         images = self.random_brightness(images)
#         return images

class Jittering(layers.Layer):
    def __init__(self, sigma: float = 0.03):
        super(Jittering, self).__init__()
        self.sigma = sigma

    def call(self, signal):
        return signal + tf.random.normal(tf.shape(signal), stddev=self.sigma)
        # if hasattr(signal, 'numpy'):
        #     return tf.convert_to_tensor(jitter(signal.numpy(), self.sigma))
        # else:
        #     return signal
    
        
class Scaling(layers.Layer):
    def __init__(self, sigma: float = 0.1):
        super(Scaling, self).__init__()
        self.sigma = sigma

    def call(self, signal):
        if hasattr(signal, 'numpy'):
            return tf.convert_to_tensor(scaling(signal.numpy(), self.sigma))
        else:        
            return signal

class TimeWarping(layers.Layer):
    def __init__(self, sigma=0.2, knot=4):
        super(TimeWarping, self).__init__()
        self.sigma = sigma
        self.knot = knot

    def call(self, signal):
        if hasattr(signal, 'numpy'):
            return tf.convert_to_tensor(time_warp(signal.numpy(), self.sigma, self.knot))
        else:
            return signal

class WindowWarping(layers.Layer):
    def __init__(self, window_ratio=0.1, scales=[0.5, 2.]):
        super(WindowWarping, self).__init__()
        self.window_ratio = window_ratio
        self.scales = scales

    def call(self, signal):
        if hasattr(signal, 'numpy'):
            return tf.convert_to_tensor(window_warp(signal.numpy(), self.window_ratio, self.scales))
        else:
            return signal


def augmenter(name):
    return keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            Jittering(sigma=0.03),
            # Scaling(sigma=0.1),
            # TimeWarping(sigma=0.2, knot=4),
            # WindowWarping(window_ratio=0.1, scales=[0.5, 2.]),
        ],
        name=name,
    )

