import tensorflow as tf


@tf.function
def random_zoom(img, label):
    scale = tf.random.uniform([], 0.85, 1.15)
    img_h, img_w, img_channels = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2]
    img_h_new, img_w_new = tf.cast(tf.cast(img_h, tf.float32) * scale, tf.int32), tf.cast(tf.cast(img_w, tf.float32) * scale, tf.int32)
    img_h_diff, img_w_diff = tf.math.abs(img_h_new - img_h), tf.math.abs(img_w_new - img_w)
    img_t, img_l = img_h_diff // 2, img_w_diff // 2
    img_b, img_r = img_h_diff - img_t, img_w_diff - img_l
    img_new = tf.image.resize(img, [img_h_new, img_w_new], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return tf.cond(scale >= 1.0,
                   lambda: img_new[img_t:img_t + img_h, img_l:img_l + img_w, :],
                   lambda: tf.pad(img_new, [[img_t, img_b], [img_l, img_r], [0, 0]])), label







