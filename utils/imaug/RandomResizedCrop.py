import tensorflow as tf


# 随机缩放裁剪
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


