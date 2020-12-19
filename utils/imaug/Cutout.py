import tensorflow as tf


@tf.function
def random_cutout(images,
                  labels,
                  mask_size,
                  constant_values=0,
                  seed=None):
    batch_size = tf.shape(images)[0]
    mask_size, image_height, image_width = _norm_params(images, mask_size)

    cutout_center_height = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32, seed=seed
    )
    cutout_center_width = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32, seed=seed
    )

    offset = tf.transpose([cutout_center_height, cutout_center_width], [1, 0])

    origin_shape = images.shape
    offset = tf.convert_to_tensor(offset)

    mask_size = mask_size // 2

    if tf.rank(offset) == 1:
        offset = tf.expand_dims(offset, 0)
    cutout_center_heights = offset[:, 0]
    cutout_center_widths = offset[:, 1]

    lower_pads = tf.maximum(0, cutout_center_heights - mask_size[0])
    upper_pads = tf.maximum(0, image_height - cutout_center_heights - mask_size[0])
    left_pads = tf.maximum(0, cutout_center_widths - mask_size[1])
    right_pads = tf.maximum(0, image_width - cutout_center_widths - mask_size[1])

    cutout_shape = tf.transpose(
        [
            image_height - (lower_pads + upper_pads),
            image_width - (left_pads + right_pads),
        ],
        [1, 0],
    )
    masks = tf.TensorArray(images.dtype, 0, dynamic_size=True)
    for i in tf.range(tf.shape(cutout_shape)[0]):
        padding_dims = [
            [lower_pads[i], upper_pads[i]],
            [left_pads[i], right_pads[i]],
        ]
        mask = tf.pad(
            tf.zeros(cutout_shape[i], dtype=images.dtype),
            padding_dims,
            constant_values=1,
        )
        masks = masks.write(i, mask)

    mask_4d = tf.expand_dims(masks.stack(), -1)
    mask = tf.tile(mask_4d, [1, 1, 1, tf.shape(images)[-1]])
    images = tf.where(
        mask == 0,
        tf.ones_like(images, dtype=images.dtype) * constant_values,
        images,
    )

    images.set_shape(origin_shape)
    return images, labels


def _norm_params(images, mask_size):
    mask_size = tf.convert_to_tensor(mask_size)
    if tf.executing_eagerly():
        tf.assert_equal(
            tf.reduce_any(mask_size % 2 != 0),
            False,
            "mask_size should be divisible by 2",
        )
    if tf.rank(mask_size) == 0:
        mask_size = tf.stack([mask_size, mask_size])
    image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]
    return mask_size, image_height, image_width

