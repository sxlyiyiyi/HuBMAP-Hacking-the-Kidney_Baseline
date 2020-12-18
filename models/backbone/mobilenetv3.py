import tensorflow as tf
from tensorflow.keras.layers import (Conv2D,
                                     DepthwiseConv2D,
                                     Dense,
                                     GlobalAveragePooling2D,
                                     Input,
                                     Activation,
                                     BatchNormalization,
                                     Reshape,
                                     Multiply,
                                     Add)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def relu6(x):
    return tf.keras.activations.relu(x, max_value=6.0)


def h_swish(x):
    return x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0


def return_activation(x, nl):
    # 用于判断使用哪个激活函数
    if nl == 'HS':
        x = Activation(h_swish)(x)
    elif nl == 'RE':
        x = Activation(relu6)(x)

    return x


def squeeze(inputs):
    inputs_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(inputs_channels / 4))(x)
    x = Activation(relu6)(x)
    x = Dense(inputs_channels)(x)
    x = Activation(h_swish)(x)
    x = Reshape((1, 1, inputs_channels))(x)
    x = Multiply()([inputs, x])

    return x


def conv_block(inputs, filters, kernel_size, strides, nl):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = return_activation(x, nl)
    return x


def bottleneck(inputs, filters, kernel_size, exp_chan, strides, sq, nl, alpha=1):
    input_shape = tf.keras.backend.int_shape(inputs)
    tchannel = int(exp_chan)
    cchannel = int(alpha * filters)
    r = strides == 1 and input_shape[3] == filters
    # 1×1卷积调整通道数，通道数上升
    x = conv_block(inputs, tchannel, 1, 1, nl)
    # 进行3×3深度可分离卷积
    x = DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = return_activation(x, nl)
    # 引入se模块
    if sq:
        x = squeeze(x)
    # 下降通道数
    x = Conv2D(cchannel, 1, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if r:
        x = Add()([x, inputs])
    return x


def MobileNetv3_small(classes, input_shape, alpha=1, include_top=False):
    inputs = Input(input_shape, name='inputs')
    # 224, 224, 3 -> 112, 112, 16
    x = conv_block(inputs, 16, 3, 2, nl='HS', )
    # 112, 112, 16 -> 56, 56, 16
    x = bottleneck(x, 16, 3, exp_chan=16, strides=2, sq=True, nl='RE', alpha=alpha)
    # 56, 56, 16 -> 28, 28, 24
    x = bottleneck(x, 24, 3, exp_chan=72, strides=2, sq=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 24, 3, exp_chan=88, strides=1, sq=False, nl='RE', alpha=alpha)
    # 28, 28, 24 -> 14, 14, 40
    x = bottleneck(x, 40, 5, exp_chan=96, strides=2, sq=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 40, 5, exp_chan=240, strides=1, sq=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 40, 5, exp_chan=240, strides=1, sq=True, nl='HS', alpha=alpha)
    # 14, 14, 40 -> 14, 14, 48
    x = bottleneck(x, 48, 5, exp_chan=120, strides=1, sq=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 48, 5, exp_chan=144, strides=1, sq=True, nl='HS', alpha=alpha)
    # 14, 14, 48 -> 7, 7, 96
    x = bottleneck(x, 96, 5, exp_chan=288, strides=2, sq=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, 5, exp_chan=576, strides=1, sq=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, 5, exp_chan=576, strides=1, sq=True, nl='HS', alpha=alpha)
    x = conv_block(x, 576, 1, 1, nl='HS')
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 576))(x)
        x = Conv2D(1024, 1, 1, 'same')(x)
        x = return_activation(x, 'HS')
        x = Conv2D(classes, 1, 1, 'same')(x)
        x = Activation(tf.nn.softmax)(x)
        x = Reshape((classes,))(x)
    return Model(inputs, x, name='MobileNetv3_small')


def MobileNetv3_large(classes, input_shape, alpha=1, include_top=False):
    inputs = Input(input_shape, name='inputs')
    # 224, 224, 3 -> 112, 112, 16
    x = conv_block(inputs, 16, 3, strides=2, nl='HS')
    x = bottleneck(x, 16, 3, exp_chan=16, strides=1, sq=False, nl='RE', alpha=alpha)
    # 112, 112, 16 -> 56, 56, 24
    x = bottleneck(x, 24, 3, exp_chan=64, strides=2, sq=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 24, 3, exp_chan=72, strides=1, sq=False, nl='RE', alpha=alpha)
    # 56, 56, 24 -> 28, 28,40
    x = bottleneck(x, 40, 5, exp_chan=72, strides=2, sq=True, nl='RE', alpha=alpha)
    x = bottleneck(x, 40, 5, exp_chan=120, strides=1, sq=True, nl='RE', alpha=alpha)
    x = bottleneck(x, 40, 5, exp_chan=120, strides=1, sq=True, nl='RE', alpha=alpha)
    # 28, 28, 40 -> 14, 14, 80
    x = bottleneck(x, 80, 3, exp_chan=240, strides=2, sq=False, nl='HS', alpha=alpha)
    x = bottleneck(x, 80, 3, exp_chan=200, strides=1, sq=False, nl='HS', alpha=alpha)
    x = bottleneck(x, 80, 3, exp_chan=184, strides=1, sq=False, nl='HS', alpha=alpha)
    x = bottleneck(x, 80, 3, exp_chan=184, strides=1, sq=False, nl='HS', alpha=alpha)
    # 14, 14, 80 -> 14, 14, 112
    x = bottleneck(x, 112, 3, exp_chan=480, strides=1, sq=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 112, 3, exp_chan=672, strides=1, sq=True, nl='HS', alpha=alpha)
    # 14, 14, 112 -> 7, 7, 160
    x = bottleneck(x, 160, 5, exp_chan=672, strides=2, sq=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 160, 5, exp_chan=960, strides=1, sq=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 160, 5, exp_chan=960, strides=1, sq=True, nl='HS', alpha=alpha)
    # 7, 7, 160 -> 7, 7, 960
    x = conv_block(x, 960, 1, 1, nl='HS')
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 960))(x)
        x = Conv2D(1280, 1, 1, 'same')(x)
        x = return_activation(x, 'HS')
        logits = Conv2D(classes, 1, 1, 'same')(x)
        x = Activation(tf.nn.softmax)(logits)
        x = Reshape((classes,))(x)
    return Model(inputs, x, name='MobileNetV3_large')


if __name__ == '__main__':
    model = MobileNetv3_small(5, (320, 320, 3), include_top=True)
    model.summary()