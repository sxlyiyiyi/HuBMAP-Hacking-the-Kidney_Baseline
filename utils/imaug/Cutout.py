import tensorflow_addons as tfa
import tensorflow as tf


@tf.function
def cutout(image, label, mask_size=100):
    image = tfa.image.random_cutout(images=image, mask_size=mask_size, constant_values=0)

    return image, label
