import tensorflow as tf
from tensorflow.keras.layers import (Conv2D,
                                     BatchNormalization,
                                     ReLU,
                                     MaxPool2D,
                                     Concatenate,
                                     GlobalAveragePooling2D,
                                     Add,
                                     DepthwiseConv2D,
                                     SeparableConv2D,
                                     AveragePooling2D,
                                     Multiply,
                                     Dropout,
                                     Input,
                                     Activation)
from tensorflow.keras.models import Model
import tensorflow_addons as tfa


# mish激活函数
def mish(x, name):
    return tf.keras.layers.Lambda(lambda x: x * tf.tanh(tf.math.log(1 + tf.exp(x))), name=name)(x)


def bnorgn(inputs, normal='BN'):
    if normal == 'BN':
        x = BatchNormalization()(inputs)
    elif normal == 'GN':
        x = tfa.layers.GroupNormalization(groups=32)(inputs)
    else:
        raise ValueError("unsupported normal type")

    return x


def conv2d_bn_relu(inputs, out_chan, kernel_size=3, stride=1, padding='same', dilation=1, use_bias=False,
                   normal='BN', depth_multiplier=0, ac=True, ac_type='relu'):
    if depth_multiplier == 0:
        x = Conv2D(out_chan, kernel_size, stride, padding, use_bias=use_bias, dilation_rate=dilation)(inputs)
    else:
        x = DepthwiseConv2D(kernel_size, stride, 'same', depth_multiplier=depth_multiplier, use_bias=use_bias)(inputs)
    if normal == 'BN':
        x = BatchNormalization()(x)
    elif normal == 'GN':
        x = tfa.layers.GroupNormalization(groups=32)(x)
    if ac:
        if ac_type == 'relu':
            x = ReLU()(x)
        elif ac_type == 'mish':
            x = mish(x)
    return x


def DetailBranch(inputs):
    s1 = conv2d_bn_relu(inputs, 64, 3, 2)
    s1 = conv2d_bn_relu(s1, 64, 3, 1)
    s2 = conv2d_bn_relu(s1, 64, 3, 2)
    s2 = conv2d_bn_relu(s2, 64, 3, 1)
    s2 = conv2d_bn_relu(s2, 64, 3, 1)
    s3 = conv2d_bn_relu(s2, 128, 3, 2)
    s3 = conv2d_bn_relu(s3, 128, 3, 1)
    s3 = conv2d_bn_relu(s3, 128, 3, 1)
    return s3


# StemBlock
def StemBlock(inputs, out_chan):
    x = conv2d_bn_relu(inputs, out_chan, 3, 2)
    # left
    left = conv2d_bn_relu(x, out_chan // 2, 1, 1)
    left = conv2d_bn_relu(left, out_chan, 3, 2)
    # right
    right = MaxPool2D(3, strides=2, padding='same')(x)
    # concat
    fuse = Concatenate(axis=-1)([left, right])
    fuse = conv2d_bn_relu(fuse, out_chan, 3, 1)
    return fuse


# CEBlock
def CEBlock(inputs, out_chan):
    x = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    # x = GlobalAveragePooling2D(name='backbone/SB/CEBlock/GloAvePool')(inputs)
    # x = tf.expand_dims(x, 1)
    # x = tf.expand_dims(x, 1)
    x = bnorgn(x)
    x = conv2d_bn_relu(x, out_chan, 1, 1)
    x = Add()([x, inputs])
    x = conv2d_bn_relu(x, out_chan, 3, 1)
    return x


# GELayerS1
def GELayerS1(inputs, out_chan, exp_ratio=6):
    mid_chan = inputs.shape[3] * exp_ratio
    x = conv2d_bn_relu(inputs, inputs.shape[3], 3, 1)
    x = conv2d_bn_relu(x, mid_chan, 3, 1, depth_multiplier=exp_ratio)
    x = conv2d_bn_relu(x, out_chan, 1, 1, ac=False)
    x = Add()([x, inputs])
    x = ReLU()(x)
    return x


# GELayerS2
def GELayerS2(inputs, out_chan, exp_ratio=6):
    mid_chan = inputs.shape[3] * exp_ratio
    x = conv2d_bn_relu(inputs, inputs.shape[3], 3, 1)
    # dw
    x = conv2d_bn_relu(x, mid_chan, 3, 2, depth_multiplier=exp_ratio, ac=False)
    x = conv2d_bn_relu(x, mid_chan, 3, 1, depth_multiplier=1)
    x = conv2d_bn_relu(x, out_chan, 1, 1, ac=False)
    # shortcut
    shortcut = conv2d_bn_relu(inputs, inputs.shape[3], 3, 2, depth_multiplier=1, ac=False)
    shortcut = conv2d_bn_relu(shortcut, out_chan, 1, 1, ac=False)
    x = Add()([x, shortcut])
    x = ReLU()(x)

    return x


# Seman Branch
def SegmentBranch(inputs):
    s1s2 = StemBlock(inputs, 16)
    s3 = GELayerS2(s1s2, 32)
    s3 = GELayerS1(s3, 32)
    s4 = GELayerS2(s3, 64)
    s4 = GELayerS1(s4, 64)
    s5 = GELayerS2(s4, 128)
    s5 = GELayerS1(s5, 128)
    s5 = GELayerS1(s5, 128)
    s5_4 = GELayerS1(s5, 128)
    s5_5 = CEBlock(s5_4, 128)
    return s1s2, s3, s4, s5_4, s5_5


# Bilateral Guided Aggregation
def BGALayer(in_d, in_s, out_chan):
    # _, h, w, _,  = in_d.get_shape().as_list()
    left1 = conv2d_bn_relu(in_d, in_d.shape[3], 3, 1, depth_multiplier=1, ac=False)
    left1 = Conv2D(out_chan, 1, 1, 'same', use_bias=False)(left1)

    left2 = conv2d_bn_relu(in_d, out_chan, 3, 2, ac=False)
    left2 = AveragePooling2D((3, 3), 2, 'same')(left2)

    right1 = conv2d_bn_relu(in_s, out_chan, 3, 1, ac=False)
    right1 = tf.compat.v1.image.resize_bilinear(right1, [tf.shape(in_d)[1], tf.shape(in_d)[2]], align_corners=True)
    right1 = tf.sigmoid(right1)

    right2 = conv2d_bn_relu(in_s, in_s.shape[3], 3, 1, depth_multiplier=1, ac=False)
    right2 = Conv2D(out_chan, 1, 1, 'same', use_bias=False, )(right2)
    right2 = tf.sigmoid(right2)

    left = Multiply()([left1, right1])
    right = Multiply()([left2, right2])
    right = tf.compat.v1.image.resize_bilinear(right, [tf.shape(in_d)[1], tf.shape(in_d)[2]], align_corners=True)

    out = Add()([left, right])
    out = conv2d_bn_relu(out, out_chan, 3, 1)
    return out


# Seg Head
def SementHead(inputs, mid_chan, n_classes, size=None, dropprob=0.,):
    feat = conv2d_bn_relu(inputs, mid_chan, 3, 1)
    feat = Dropout(dropprob)(feat)
    feat = Conv2D(n_classes, 1, 1, 'same')(feat)

    feat = tf.compat.v1.image.resize_bilinear(feat, size, align_corners=True)
    return feat


def BiseNetV2(n_classes,
              size,
              channel=3,
              dropprob=0,
              activation=None,
              output_aux=True):
    x = inputs = Input([size[0], size[1], channel])
    db = DetailBranch(x)
    sb2, sb3, sb4, sb5_4, sb_s = SegmentBranch(x)
    head = BGALayer(db, sb_s, 128)
    logits = SementHead(head, 1024, n_classes, [tf.shape(x)[1], tf.shape(x)[2]], dropprob=dropprob)
    if activation in {'softmax', 'sigmoid'}:
        logits = Activation(activation, name=activation)(logits)

    if output_aux:
        logits_aux2 = SementHead(sb2, 128, n_classes, [tf.shape(x)[1], tf.shape(x)[2]], dropprob=dropprob)
        logits_aux3 = SementHead(sb3, 128, n_classes, [tf.shape(x)[1], tf.shape(x)[2]], dropprob=dropprob)
        logits_aux4 = SementHead(sb4, 128, n_classes, [tf.shape(x)[1], tf.shape(x)[2]], dropprob=dropprob)
        logits_aux5_4 = SementHead(sb5_4, 128, n_classes, [tf.shape(x)[1], tf.shape(x)[2]], dropprob=dropprob)

        if activation in {'softmax', 'sigmoid'}:
            logits_aux2 = Activation(activation)(logits_aux2)
            logits_aux3 = Activation(activation)(logits_aux3)
            logits_aux4 = Activation(activation)(logits_aux4)
            logits_aux5_4 = Activation(activation)(logits_aux5_4)
        return Model(inputs, (logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4), name="BiseNetV2")

    return Model(inputs, logits, name='BiseNetV2')


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    import os

    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

    model = BiseNetV2(19, [None, None])
    model.summary()
    plot_model(model, to_file='./bisenetv2.png', show_shapes=True)











