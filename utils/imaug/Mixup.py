import tensorflow as tf


def mixup(image, label, num_classes, probability=1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    batch_size, img_h, img_w = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]

    imgs = []
    labs = []
    for j in range(batch_size):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= probability, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
        a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j, ]
        img2 = image[k, ]
        imgs.append((1 - a) * img1 + a * img2)
        # MAKE CUTMIX LABEL
        if len(label.shape) == 1:
            lab1 = tf.one_hot(label[j], num_classes)
            lab2 = tf.one_hot(label[k], num_classes)
        else:
            lab1 = label[j, ]
            lab2 = label[k, ]
        labs.append((1 - a) * lab1 + a * lab2)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs), (batch_size, img_h, img_w, 3))
    label2 = tf.reshape(tf.stack(labs), (batch_size, num_classes))
    return image2, label2
