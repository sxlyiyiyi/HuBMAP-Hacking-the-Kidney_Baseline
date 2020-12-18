import tensorflow as tf


# CutMix, 在train里使用
@tf.function
def cutmix(img_batch, label_batch, batch_size, num_classes):
    """

    :param img_batch:
    :param label_batch: 非 one_hot 编码
    :param batch_size:
    :param num_classes:
    :return: one_hot 编码
    """
    if len(label_batch.shape) == 1:
        label_batch = tf.one_hot(label_batch, num_classes, axis=-1)
    img_h, img_w, img_channels = tf.shape(img_batch)[1], tf.shape(img_batch)[2], tf.shape(img_batch)[3]
    imgs, labs = [], []
    # CHOOSE RANDOM LOCATION
    cut_xs = tf.cast(tf.random.uniform([batch_size], 0, tf.cast(img_w, tf.float32)), tf.int32)
    cut_ys = tf.cast(tf.random.uniform([batch_size], 0, tf.cast(img_h, tf.float32)), tf.int32)
    cut_ratios = tf.math.sqrt(1 - tf.random.uniform([batch_size], 0, 1))  # cut ratio
    cut_ws = tf.cast(tf.cast(img_w, tf.float32) * cut_ratios, tf.int32)
    cut_hs = tf.cast(tf.cast(img_h, tf.float32) * cut_ratios, tf.int32)
    yas = tf.math.maximum(0, cut_ys - cut_hs // 2)
    ybs = tf.math.minimum(img_h, cut_ys + cut_hs // 2)
    xas = tf.math.maximum(0, cut_xs - cut_ws // 2)
    xbs = tf.math.minimum(img_w, cut_xs + cut_ws // 2)
    # CHOOSE RANDOM IMAGE TO CUTMIX WITH
    js = tf.random.shuffle(tf.range(batch_size, dtype=tf.int32))
    for i in range(batch_size):
        ya, yb, xa, xb, j = yas[i], ybs[i], xas[i], xbs[i], js[i]
        img_org, img_cut, label_org, label_cut = img_batch[i], img_batch[j], label_batch[i], label_batch[j]
        # MAKE CUTMIX IMAGE
        img_org_left_middle = img_org[ya:yb, 0:xa, :]
        img_cut = img_cut[ya:yb, xa:xb, :]
        img_org_right_middle = img_org[ya:yb, xb:img_w, :]
        img_middle = tf.concat([img_org_left_middle, img_cut, img_org_right_middle], axis=1)
        img_cutmix = tf.concat([img_org[0:ya, :, :], img_middle, img_org[yb:img_h, :, :]], axis=0)
        imgs.append(img_cutmix)

        # MAKE CUTMIX LABEL
        label_weight = tf.cast((yb - ya) * (xb - xa) / (img_h * img_w), tf.float32)
        label_cutmix = (1 - label_weight) * label_org + label_weight * label_cut
        labs.append(label_cutmix)

    img_batch = tf.reshape(tf.stack(imgs), (batch_size, img_h, img_w, 3))
    label_batch = tf.reshape(tf.stack(labs), (batch_size, num_classes))

    return img_batch, label_batch
