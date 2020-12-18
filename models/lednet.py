import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import (Conv2D,
                                     Activation,
                                     BatchNormalization,
                                     ReLU,
                                     Dropout,
                                     Add,
                                     MaxPool2D,
                                     GlobalAveragePooling2D,
                                     Lambda,
                                     Multiply,
                                     Input,
                                     AveragePooling2D)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model


# 通道分离和混洗
def _channel_shuffle(x, groups):
    batch_szie, height, width, num_channels = x.get_shape().as_list()

    channels_per_group = num_channels // groups

    # reshape
    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])

    # flatten
    x = tf.reshape(x, [-1, height, width, num_channels])

    return x


# split-shuffle-bottleneck(SS-bt)
def SS_nbt_module(inputs, dilated, channels, dropprob):
    """

    :param inputs:
    :param dilated:
    :param channels:
    :param dropprob:
    :return:
    """
    # 通道分离
    oup_inc = channels//2
    residual = inputs
    x1, x2 = tf.split(inputs, 2, axis=3)

    # 第一个通道
    x1 = Conv2D(oup_inc, [3, 1], 1, padding='same', kernel_regularizer=l2(0.0001))(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(oup_inc, [1, 3], 1, padding='same', kernel_regularizer=l2(0.0001))(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    x1 = Conv2D(oup_inc, [3, 1], 1, padding='same', dilation_rate=(dilated, 1), kernel_regularizer=l2(0.0001))(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(oup_inc, [1, 3], 1, padding='same', dilation_rate=(1, dilated), kernel_regularizer=l2(0.0001))(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    # 第二个通道
    x2 = Conv2D(oup_inc, [1, 3], 1, padding='same', kernel_regularizer=l2(0.0001))(x2)
    x2 = ReLU()(x2)
    x2 = Conv2D(oup_inc, [3, 1], 1, padding='same', kernel_regularizer=l2(0.0001))(x2)
    x2 = BatchNormalization()(x2)

    x2 = Conv2D(oup_inc, [1, 3], 1, padding='same', dilation_rate=(1, dilated), kernel_regularizer=l2(0.0001))(x2)
    x2 = ReLU()(x2)
    x2 = Conv2D(oup_inc, [3, 1], 1, padding='same', dilation_rate=(dilated, 1), kernel_regularizer=l2(0.0001))(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    if dropprob != 0:
        x1 = Dropout(rate=dropprob)(x1)
        x2 = Dropout(rate=dropprob)(x2)

    # 连接两个group
    output = tf.concat([x1, x2], axis=-1)
    output = Add()([residual, output])
    output = ReLU()(output)
    # 通道混洗
    output = _channel_shuffle(output, 2)

    return output


# 下采样
def downsampler_block(inputs, in_channel, out_channel):
    # 两个并行输出
    x1 = MaxPool2D(2, 2, padding='same')(inputs)
    x2 = Conv2D(out_channel - in_channel, [3, 3], 2, 'same', kernel_regularizer=l2(0.0001))(inputs)

    # 叠加两个并行输出
    diffY = x2.shape[1] - x1.shape[1]
    diffX = x2.shape[2] - x1.shape[2]

    x1 = tf.pad(x1, [[0, 0], [diffX // 2, diffX - diffX // 2],
                     [diffY // 2, diffY - diffY // 2], [0, 0]])

    output = tf.concat([x2, x1], axis=-1)
    output = BatchNormalization()(output)
    output = ReLU()(output)
    return output


# 编码器
def encoder(inputs):
    # 下采样模块
    x = downsampler_block(inputs, in_channel=3, out_channel=32)
    # SS-nbt unit × 3
    x = SS_nbt_module(x, dilated=1, channels=32, dropprob=0.03)
    x = SS_nbt_module(x, dilated=1, channels=32, dropprob=0.03)
    x = SS_nbt_module(x, dilated=1, channels=32, dropprob=0.03)
    # 下采样模块
    x = downsampler_block(x, in_channel=32, out_channel=64)
    # SS-nbt unit × 2
    x = SS_nbt_module(x, dilated=1, channels=64, dropprob=0.03)
    x = SS_nbt_module(x, dilated=1, channels=64, dropprob=0.03)
    # 下采样模块
    x = downsampler_block(x, in_channel=64, out_channel=128)
    # SS-nbt × 8
    x = SS_nbt_module(x, dilated=1, channels=128, dropprob=0.3)
    x = SS_nbt_module(x, dilated=2, channels=128, dropprob=0.3)
    x = SS_nbt_module(x, dilated=5, channels=128, dropprob=0.3)
    x = SS_nbt_module(x, dilated=9, channels=128, dropprob=0.3)
    x = SS_nbt_module(x, dilated=2, channels=128, dropprob=0.3)
    x = SS_nbt_module(x, dilated=5, channels=128, dropprob=0.3)
    x = SS_nbt_module(x, dilated=9, channels=128, dropprob=0.3)
    x = SS_nbt_module(x, dilated=17, channels=128, dropprob=0.3)

    return x


# Batch_Relu
def conv2d_bn_relu(inputs, channels, kernel_size=1, stride=1, padding='same'):
    x = Conv2D(channels, kernel_size, stride, padding, kernel_regularizer=l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


# APN注意力金字塔
def apn_module_simplified(inputs, out_ch):
    b, h, w, c = inputs.get_shape().as_list()
    # branch1
    # branch1 = Lambda(lambda x: tf.reduce_mean(x, [1, 2], keepdims=True, name='global_average_pooling'))(inputs)
    branch1 = GlobalAveragePooling2D()(inputs)
    branch1 = tf.expand_dims(branch1, 1)
    branch1 = tf.expand_dims(branch1, 1)

    branch1 = conv2d_bn_relu(branch1, out_ch)

    branch1 = Lambda(lambda x: tf.image.resize(x, [h, w]))(branch1)

    # mid
    branch2 = conv2d_bn_relu(inputs, out_ch)

    # branch3
    x1 = conv2d_bn_relu(inputs, 1, kernel_size=7, stride=2, padding='same')
    x2 = conv2d_bn_relu(x1, 1, kernel_size=5, stride=2, padding='same')

    x3 = conv2d_bn_relu(x2, 1, kernel_size=3, stride=2, padding='same')
    x3 = conv2d_bn_relu(x3, 1, kernel_size=3, stride=1, padding='same')
    x3 = Lambda(lambda x: tf.image.resize(x, [np.ceil(h/4).astype(np.int32), np.ceil(w/4).astype(np.int32)]))(x3)

    x2 = conv2d_bn_relu(x2, 1, kernel_size=5, stride=1, padding='same')

    x = Add()([x2, x3])
    x = Lambda(lambda x: tf.image.resize(x, [h//2, w//2]))(x)

    x1 = conv2d_bn_relu(x1, 1, kernel_size=7, stride=1, padding='same')
    x = Add()([x, x1])
    x = Lambda(lambda x: tf.image.resize(x, [h, w]))(x)

    # 加权融合
    x = Multiply()([x, branch2])
    # point-wise sum
    x = Add()([x, branch1])

    return x


def apn_module(x_in, out_ch, name=None):
    b, h, w, c = x_in.get_shape().as_list()
    # 第一个分支，全局平均池化来获取全局上下文
    branch1 = GlobalAveragePooling2D()(x_in)
    branch1 = tf.expand_dims(branch1, axis=1)
    branch1 = tf.expand_dims(branch1, axis=1)
    branch1 = conv2d_bn_relu(branch1, out_ch)
    branch1 = Lambda(lambda x: tf.image.resize(x, [h, w]))(branch1)
    # 第二个分支, 1×1卷积
    branch2 = conv2d_bn_relu(x_in, out_ch, kernel_size=1)
    # 第三个分支
    x1 = conv2d_bn_relu(x_in, 128, kernel_size=7, stride=2)
    x2 = conv2d_bn_relu(x1, 128, kernel_size=5, stride=2)
    x3 = conv2d_bn_relu(x2, 128, kernel_size=3, stride=2)
    x3 = conv2d_bn_relu(x3, out_ch, kernel_size=1, stride=1)
    x3 = Lambda(lambda x: tf.image.resize(x, [np.ceil(h/4).astype(np.int32), np.ceil(w/4).astype(np.int32)]))(x3)
    x2 = conv2d_bn_relu(x2, out_ch, kernel_size=1, stride=1)
    # point-wise sum
    x = Add()([x3, x2])
    x = Lambda(lambda x: tf.image.resize(x, [h//2, w//2]))(x)
    x1 = conv2d_bn_relu(x1, out_ch, kernel_size=1, stride=1)
    # point-wise sum
    x = Add()([x, x1])
    x = Lambda(lambda x: tf.image.resize(x, [h, w]))(x)
    # point-wise product
    x = Multiply()([x, branch2])
    # # point-wise sum
    x = Add()([x, branch1])
    return x


# 解码层
def decode_s(inputs, img_height, img_width, num_classes):
    apn = apn_module_simplified(inputs, num_classes)
    decode_output = Lambda(lambda x: tf.image.resize(x, [img_height, img_width]))(apn)
    return decode_output


# 解码层
def decode(inputs, img_height, img_width, out_channel=2):
    apn = apn_module(inputs, out_channel)
    decode_output = Lambda(lambda x: tf.image.resize(x, [img_height, img_width]), name='decode_output')(apn)
    return decode_output


# LEDNet网络结构
def LEDNet_S(num_class, img_height, img_width, channel=3, activation='softmax', cls_aux=False):
    x = inputs = Input([img_height, img_width, channel])
    encode_output = encoder(x)
    decode_output = decode_s(encode_output, img_height, img_width, num_class)
    if cls_aux:
        cls_result = Conv2D(num_class, [3, 3], strides=2)(encode_output)
        cls_result = Lambda(lambda x: tf.reduce_mean(x, axis=1))(cls_result)
        cls_result = Lambda(lambda x: tf.reduce_mean(x, axis=1))(cls_result)
        if activation in {'softmax', 'sigmoid'}:
            decode_output = Activation(activation)(decode_output)
            cls_result = Activation(activation)(cls_result)
        return Model(inputs, (decode_output, cls_result), name="LEDNet")
    if activation in {'softmax', 'sigmoid'}:
        decode_output = Activation(activation)(decode_output)

    return Model(inputs, decode_output, name="LEDNet_S")


# LEDNet网络结构
def LEDNet(num_class, img_height, img_width, channel=3, activation='softmax', cls_aux=False):
    x = inputs = Input([img_height, img_width, channel])
    encode_output = encoder(x)
    decode_output = decode(encode_output, img_height, img_width)
    if cls_aux:

        cls_result = Conv2D(num_class, [3, 3], strides=2)(encode_output)
        cls_result = Lambda(lambda x: tf.reduce_mean(x, axis=1))(cls_result)
        cls_result = Lambda(lambda x: tf.reduce_mean(x, axis=1), name='cls_result')(cls_result)
        if activation in {'softmax', 'sigmoid'}:
            decode_output = Activation(activation)(decode_output)
            cls_result = Activation(activation)(cls_result)
        return Model(inputs, (decode_output, cls_result), name="LEDNet")
    if activation in {'softmax', 'sigmoid'}:
        decode_output = Activation(activation)(decode_output)

    return Model(inputs, decode_output, name="LEDNet")


# 损失函数
def LEDLoss():
    def led_loss(mask_y_true, mask_y_pred, cls_true, cls_pred):
        # cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(mask_y_true, mask_y_pred, from_logits=True)
        # cross_entropy_cls = tf.keras.losses.sparse_categorical_crossentropy(cls_true, cls_pred, from_logits=True)
        # cross_entropy = tf.reduce_mean(cross_entropy)
        # cross_entropy_cls = tf.reduce_mean(cross_entropy_cls)
        mask_y_true = tf.cast(tf.squeeze(mask_y_true), dtype=tf.int32)
        # mask_y_true = tf.cast(mask_y_true[:, :, :, 0], dtype=tf.int32)
        cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=mask_y_pred, labels=mask_y_true)
        cross_entropy_cls = tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=cls_pred, labels=cls_true)
        return cross_entropy, cross_entropy_cls
    return led_loss


if __name__ == "__main__":

    pass










