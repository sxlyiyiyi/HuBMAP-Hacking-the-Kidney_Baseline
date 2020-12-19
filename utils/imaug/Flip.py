import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


# 随机翻转
@tf.function
def random_flip(img, mask, vertical=False):
    img, mask = random_flip_left_right(img, mask)
    if vertical is True:
        img, mask = random_flip_up_down(img, mask)

    return img, mask


def random_flip_up_down(image, mask, seed=None):
    return _random_flip(image, mask, 0, seed, 'random_flip_up_down')


def random_flip_left_right(image, mask, seed=None):
    return _random_flip(image, mask, 1, seed, 'random_flip_left_right')


def _random_flip(image, mask, flip_index, seed, scope_name):
    """Randomly (50% chance) flip an image along axis `flip_index`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    flip_index: Dimension along which to flip the image.
      Vertical: 0, Horizontal: 1
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.
    scope_name: Name of the scope in which the ops are added.

  Returns:
    A tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
    with ops.name_scope(None, scope_name, [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        shape = image.get_shape()
        if shape.ndims == 3 or shape.ndims is None:
            uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
            mirror_cond = math_ops.less(uniform_random, .5)
            img_result, mask_result = control_flow_ops.cond(
                mirror_cond,
                lambda: (array_ops.reverse(image, [flip_index]), array_ops.reverse(mask, [flip_index])),
                lambda: (image, mask),
                name=scope)
            return img_result, mask_result
        else:
            raise ValueError(
                '\'image\' (shape %s) must have 3 dimensions.' % shape)
