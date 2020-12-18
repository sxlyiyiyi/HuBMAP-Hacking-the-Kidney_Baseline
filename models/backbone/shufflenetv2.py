import tensorflow as tf
from tensorflow.keras.layers import (Conv2D,
                                     BatchNormalization,
                                     Lambda,
                                     Activation,
                                     DepthwiseConv2D,
                                     Concatenate,
                                     Input,
                                     MaxPool2D,
                                     GlobalAveragePooling2D,
                                     Dense)
from tensorflow.keras.regularizers import l2
import numpy as np


# 通道分离
def channel_split(x, name=''):
    in_channels = x.get_shape().as_list()[-1]
    ip = in_channels // 2
    # 对通道进行分割
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name=f'{name}/sp0_slice')(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name=f'{name}/sp1_slice')(x)

    return c_hat, c


# 通道混洗
def channel_shuffle(x):
    batch_szie, height, width, num_channels = x.get_shape().as_list()

    channels_per_group = num_channels // 2

    # reshape
    x = tf.reshape(x, [-1, height, width, 2, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])

    # flatten
    x = tf.reshape(x, [-1, height, width, num_channels])

    return x


def shuffle_unit(inputs, out_chan, bottleneck_ratio, strides=2, stage=1, block=1, weight_decay=1e-4, use_bias=False):
    prefix = f'stage{stage}/block{block}'
    bottleneck_chan = int(out_chan * bottleneck_ratio / 2)

    if strides < 2:
        c_hat, c = channel_split(inputs, name=f'{prefix}/spl')
        inputs = c

    # 右侧1×1卷积
    x = Conv2D(bottleneck_chan, 1, 1, 'same', use_bias=use_bias, kernel_regularizer=l2(weight_decay),
               name=f'{prefix}/1x1conv_1')(inputs)
    x = BatchNormalization(name=f'{prefix}/bn_1x1conv_1')(x)
    x = Activation('relu', name=f'{prefix}/relu_1x1conv_1')(x)

    # 右侧3×3深度可分离卷积
    x = DepthwiseConv2D(3, strides, 'same', use_bias=use_bias, kernel_regularizer=l2(weight_decay),
                        name=f'{prefix}/3x3dwconv')(x)
    x = BatchNormalization(name=f'{prefix}/bn_3x3dwconv')(x)

    # 右侧1×1卷积
    x = Conv2D(bottleneck_chan, 1, 1, 'same', use_bias=use_bias, kernel_regularizer=l2(weight_decay),
               name=f'{prefix}/1x1conv_2')(x)
    x = BatchNormalization(name=f'{prefix}/bn_1x1conv_2')(x)
    x = Activation('relu', name=f'{prefix}/relu_1x1conv_2')(x)

    # 当strides等于2的时候，残差边需要添加卷积
    if strides < 2:
        ret = Concatenate(name=f'{prefix}/concat_1')([x, c_hat])
    else:
        s2 = DepthwiseConv2D(3, 2, 'same', use_bias=use_bias, depthwise_regularizer=l2(weight_decay),
                             name=f'{prefix}/3x3conv-2')(inputs)
        s2 = BatchNormalization(name=f'{prefix}/bn_3x3dwconv_2')(s2)

        s2 = Conv2D(bottleneck_chan, 1, 1, 'same', use_bias=use_bias, kernel_regularizer=l2(weight_decay),
                    name=f'{prefix}/1x1_conv_3')(s2)
        s2 = BatchNormalization(name=f'{prefix}/bn_1x1_conv_3')(s2)
        s2 = Activation('relu', name=f'{prefix}/relu_1x1_conv_3')(s2)

        ret = Concatenate(name=f'p{prefix}/concat_2')([x, s2])

    ret = Lambda(channel_shuffle, name=f'{prefix}/channel_shuffle')(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1, weight_decay=1e-4, use_bias=False):
    x = shuffle_unit(x, out_chan=channel_map[stage - 1], strides=2, bottleneck_ratio=bottleneck_ratio, stage=stage,
                     block=1, weight_decay=weight_decay, use_bias=use_bias)

    for i in range(1, repeat):
        x = shuffle_unit(x, out_chan=channel_map[stage - 1], strides=1, bottleneck_ratio=bottleneck_ratio,
                         stage=stage, block=(1 + i), weight_decay=weight_decay, use_bias=use_bias)

    return x


def ShuffleNetV2(input_tensor=None,
                 pooling='max',
                 input_shape=(224, 224, 3),
                 num_shuffle_units=[3, 7, 3],
                 scale_factor=1,
                 bottleneck_ratio=1,
                 classes=1000,
                 use_bias=False,
                 weight_decay=1e-4):

    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))

    # 不同缩放比例对应的stage2的out_channel
    out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}
    # 每个stage与stage的比例
    out_channels_in_stage = np.array([1, 1, 2, 4])
    out_channels_in_stage *= out_dim_stage_two[scale_factor]  # calculate output channels for each stage
    # stage1的out_channels都是24
    out_channels_in_stage[0] = 24
    # out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is not None:
        img_input = input_tensor
    else:
        img_input = Input(shape=input_shape)

    # stage1
    x = Conv2D(out_channels_in_stage[0], 3, 2, 'same', activation='relu', kernel_regularizer=l2(weight_decay),
               name='conv1')(img_input)
    x = MaxPool2D(3, 2, 'same', name='maxpool1')(x)

    # stage2-4
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage, bottleneck_ratio, repeat=repeat, stage=stage + 2, weight_decay=weight_decay,
                  use_bias=use_bias)

    if scale_factor != 2:
        x = Conv2D(1024, 1, 1, 'same', activation='relu', kernel_regularizer=l2(weight_decay), name='1x1conv5_out')(x)
    else:
        x = Conv2D(2048, 1, 1, 'same', activation='relu', kernel_regularizer=l2(weight_decay), name='1x1conv5_out')(x)

    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    logits = Dense(classes, name='fc')(x)
    x = Activation('softmax', name='softmax')(logits)
    x = Lambda(lambda z: tf.argmax(z, axis=-1), name='pred_lab')(x)

    inputs = img_input

    model = tf.keras.models.Model(inputs, (x, logits), name=name)

    return model


if __name__ == '__main__':
    inputs1 = Input([224, 224, 3])
    model1 = ShuffleNetV2(input_tensor=inputs1, classes=1000)
    model1.summary()