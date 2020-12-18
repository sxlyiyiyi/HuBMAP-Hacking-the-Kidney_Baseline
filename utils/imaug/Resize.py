import tensorflow as tf
import numpy as np


# 随机缩放, 随机裁剪
# @tf.function
def random_resizedcrop(img, label, size, dl=0.25, ul=2.):
    scale = tf.random.uniform([], dl, ul)

    img_h, img_w, img_channels = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2]
    img_h_new, img_w_new = tf.cast(tf.cast(img_h, tf.float32) * scale, tf.int32), tf.cast(tf.cast(img_w, tf.float32) * scale, tf.int32)
    img_new = tf.image.resize(img, [img_h_new, img_w_new], method=tf.image.ResizeMethod.BILINEAR)
    label_new = tf.image.resize(label, [img_h_new, img_w_new], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    pad_h, pad_w = 0, 0
    if img_h_new == size[0] and img_w_new == size[1]:
        return img_new, label_new
    if img_h_new < size[0]:
        pad_h = (size[0] - img_h_new) // 2 + 1
    if img_w_new < size[1]:
        pad_w = (size[1] - img_w_new) // 2 + 1
    if pad_h > 0 or pad_w > 0:
        img_new = tf.pad(img_new, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]])
        label_new = tf.pad(label_new, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], 'CONSTANT', constant_values=255)
    im_h, im_w = tf.cast(tf.shape(img_new)[0], tf.float32), tf.cast(tf.shape(img_new)[1], tf.float32)
    sh, sw = tf.random.uniform([], 0, 1), tf.random.uniform([], 0, 1)
    sh, sw = tf.cast(sh * (im_h - size[0]), tf.int32), tf.cast(sw * (im_w - size[1]), tf.int32)

    img_new = img_new[sh:sh + size[0], sw:sw + size[1], :]
    label_new = label_new[sh:sh + size[0], sw:sw + size[1]]

    return img_new, label_new


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """
    在一定范围内得到随机值，范围为min_scale_factor到max_scale_factor，间隔为step_size

    Args：
        min_scale_factor(float): 随机尺度下限，大于0
        max_scale_factor(float): 随机尺度上限，不小于下限值
        step_size(float): 尺度间隔，非负, 等于为0时直接返回min_scale_factor到max_scale_factor范围内任一值

    Returns：
        随机尺度值

    """

    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return min_scale_factor

    if step_size == 0:
        return tf.random.uniform([], min_scale_factor, max_scale_factor)

    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = np.linspace(min_scale_factor, max_scale_factor,
                                num_steps).tolist()
    scale_factors = tf.random.shuffle(scale_factors)
    return scale_factors[0]


# 随机按一定步长缩放, 随机裁剪
def StepScaleCrop(img, mask, size, dl=0.25, ul=2., step_size=0.1):
    scale = get_random_scale(dl, ul, step_size)
    img_h, img_w = size[0], size[1]
    img_h_new, img_w_new = int(img_h * scale), int(img_w * scale)
    img_new = tf.image.resize(img, [img_h_new, img_w_new], method=tf.image.ResizeMethod.BILINEAR)
    mask_new = tf.image.resize(mask, [img_h_new, img_w_new], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    pad_h, pad_w = 0, 0
    if img_h_new == size[0] and img_w_new == size[1]:
        return img_new, mask_new
    if img_h_new < size[0]:
        pad_h = (size[0] - img_h_new) // 2 + 1
    if img_w_new < size[1]:
        pad_w = (size[1] - img_w_new) // 2 + 1
    if pad_h > 0 or pad_w > 0:
        img_new = tf.pad(img_new, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]])
        mask_new = tf.pad(mask_new, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], 'CONSTANT', constant_values=255)
    im_h, im_w = tf.cast(tf.shape(img_new)[0], tf.float32), tf.cast(tf.shape(img_new)[1], tf.float32)
    sh, sw = tf.random.uniform([], 0, 1), tf.random.uniform([], 0, 1)
    sh, sw = tf.cast(sh * (im_h - size[0]), tf.int32), tf.cast(sw * (im_w - size[1]), tf.int32)

    img_new = img_new[sh:sh + size[0], sw:sw + size[1], :]
    mask_new = mask_new[sh:sh + size[0], sw:sw + size[1]]

    return img_new, mask_new


# 随机缩放, 居中裁剪
@tf.function
def random_zoom(img, mask, dl=0.25, ul=2.):
    scale = tf.random.uniform([], dl, ul)
    img_h, img_w, img_channels = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2]
    img_h_new, img_w_new = tf.cast(tf.cast(img_h, tf.float32) * scale, tf.int32), tf.cast(tf.cast(img_w, tf.float32) * scale, tf.int32)
    img_h_diff, img_w_diff = tf.math.abs(img_h_new - img_h), tf.math.abs(img_w_new - img_w)
    img_t, img_l = img_h_diff // 2, img_w_diff // 2
    img_b, img_r = img_h_diff - img_t, img_w_diff - img_l
    img_new = tf.image.resize(img, [img_h_new, img_w_new], method=tf.image.ResizeMethod.BILINEAR)
    mask_new = tf.image.resize(mask, [img_h_new, img_w_new], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return tf.cond(scale >= 1.0,
                   lambda: (img_new[img_t:img_t + img_h, img_l:img_l + img_w, :],
                            mask_new[img_t:img_t + img_h, img_l:img_l + img_w, :]),
                   lambda: (tf.pad(img_new, [[img_t, img_b], [img_l, img_r], [0, 0]]),
                            tf.pad(mask_new, [[img_t, img_b], [img_l, img_r], [0, 0]], 'CONSTANT', constant_values=255)))
