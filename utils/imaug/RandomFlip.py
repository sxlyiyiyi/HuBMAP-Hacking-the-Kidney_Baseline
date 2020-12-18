import tensorflow as tf


@tf.function
def random_flip(image, label, vertical=False):
    image = tf.image.random_flip_left_right(image)
    if vertical is True:
        image = tf.image.random_flip_up_down(image)

    return image, label
