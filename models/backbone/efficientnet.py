import tensorflow as tf
from tensorflow.keras.layers import (Conv2D,
                                     BatchNormalization,
                                     Activation,
                                     ZeroPadding2D,
                                     DepthwiseConv2D,
                                     GlobalAveragePooling2D,
                                     Reshape,
                                     multiply,
                                     Dropout,
                                     add,
                                     Input,
                                     Dense,
                                     GlobalMaxPooling2D)
from tensorflow.keras.regularizers import l2
from copy import deepcopy

BASE_WEIGHTS_PATH = (
    'https://github.com/Callidior/keras-applications/'
    'releases/download/efficientnet/')

IMAGENET_WEIGHTS_HASHES = {
    'efficientnet-b0': ('163292582f1c6eaca8e7dc7b51b01c61'
                        '5b0dbc0039699b4dcd0b975cc21533dc',
                        'c1421ad80a9fc67c2cc4000f666aa507'
                        '89ce39eedb4e06d531b0c593890ccff3'),
    'efficientnet-b1': ('d0a71ddf51ef7a0ca425bab32b7fa7f1'
                        '6043ee598ecee73fc674d9560c8f09b0',
                        '75de265d03ac52fa74f2f510455ba64f'
                        '9c7c5fd96dc923cd4bfefa3d680c4b68'),
    'efficientnet-b2': ('bb5451507a6418a574534aa76a91b106'
                        'f6b605f3b5dde0b21055694319853086',
                        '433b60584fafba1ea3de07443b74cfd3'
                        '2ce004a012020b07ef69e22ba8669333'),
    'efficientnet-b3': ('03f1fba367f070bd2545f081cfa7f3e7'
                        '6f5e1aa3b6f4db700f00552901e75ab9',
                        'c5d42eb6cfae8567b418ad3845cfd63a'
                        'a48b87f1bd5df8658a49375a9f3135c7'),
    'efficientnet-b4': ('98852de93f74d9833c8640474b2c698d'
                        'b45ec60690c75b3bacb1845e907bf94f',
                        '7942c1407ff1feb34113995864970cd4'
                        'd9d91ea64877e8d9c38b6c1e0767c411'),
    'efficientnet-b5': ('30172f1d45f9b8a41352d4219bf930ee'
                        '3339025fd26ab314a817ba8918fefc7d',
                        '9d197bc2bfe29165c10a2af8c2ebc675'
                        '07f5d70456f09e584c71b822941b1952'),
    'efficientnet-b6': ('f5270466747753485a082092ac9939ca'
                        'a546eb3f09edca6d6fff842cad938720',
                        '1d0923bb038f2f8060faaf0a0449db4b'
                        '96549a881747b7c7678724ac79f427ed'),
    'efficientnet-b7': ('876a41319980638fa597acbbf956a82d'
                        '10819531ff2dcb1a52277f10c7aefa1a',
                        '60b56ff3a8daccc8d96edfd40b204c11'
                        '3e51748da657afd58034d54d3cec2bac')
}

NS_WEIGHTS_PATH = 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/'
NS_WEIGHTS_HASHES = {
    'efficientnet-b0': ('5e376ca93bc6ba60f5245d13d44e4323', 'a5b48ae7547fc990c7e4f3951230290d'),
    'efficientnet-b1': ('79d29151fdaec95ac78e1ca97fc09634', '4d35baa41ca36f175506a33918f7e334'),
    'efficientnet-b2': ('8c643222ffb73a2bfdbdf90f2cde01af', 'e496e531f41242598288ff3a4b4199f9'),
    'efficientnet-b3': ('3b29e32602dad75d1f575d9ded00f930', '47da5b154de1372b557a65795d3e6135'),
    'efficientnet-b4': ('c000bfa03bf3c93557851b4e1fe18f51', '47c10902a4949eec589ab92fe1c35ed8'),
    'efficientnet-b5': ('8a920cd4ee793f53c251a1ecd3a5cee6', '4d53ef3544d4114e2d8080d6d777a74c'),
    'efficientnet-b6': ('cc69df409516ab57e30e51016326853e', '71f96d7e15d9f891f3729b4f4e701f77'),
    'efficientnet-b7': ('1ac825752cbc26901c8952e030ae4dd9', 'e112b00c464fe929b821edbb35d1af55')
}

# 每个Blocks的参数
DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

# 两个Kernel的初始化器
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def block(inputs, activation=tf.nn.swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True, weight_decay=1e-4):
    """

    :param inputs: 输入
    :param activation: 激活函数
    :param drop_rate: dropout概率
    :param name: name
    :param filters_in: 输入通道数
    :param filters_out: 输出通道数
    :param kernel_size: kernel size
    :param strides: 步长
    :param expand_ratio: 通道数扩张倍数
    :param se_ratio: 激活比例
    :param id_skip: 是否进行跳跃连接
    :param weight_decay
    :return:
    """
    bn_axis = 3

    # 扩张通道数
    mid_chan = filters_in * expand_ratio
    # 是否需要进行扩张
    # 1×1卷积进行升维
    if expand_ratio != 1:
        x = Conv2D(mid_chan, 1, 1, padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER,
                   kernel_regularizer=l2(weight_decay), name=name + 'expand_conv')(inputs)
        x = BatchNormalization(axis=-1, name=name + 'expand_bn')(x)
        x = Activation(activation, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # if strides == 2:
    #     x = ZeroPadding2D(padding=tf.keras.applications.)

    # part2 dwconv
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', use_bias=False,
                        depthwise_initializer=CONV_KERNEL_INITIALIZER,
                        kernel_regularizer=l2(weight_decay), name=name + 'dwconv')(x)
    x = BatchNormalization(axis=-1, name=name + 'bn')(x)
    x = Activation(activation, name=name + 'activation')(x)

    # Squeeze and Excitation SE模块
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        se = Reshape((1, 1, mid_chan), name=name + 'se_reshape')(se)
        se = Conv2D(filters_se, 1, 1, 'same', activation=activation, kernel_initializer=CONV_KERNEL_INITIALIZER,
                    kernel_regularizer=l2(weight_decay), name=name + 'se_reduce')(se)
        se = Conv2D(mid_chan, 1, 1, 'same', activation='sigmoid', kernel_initializer=CONV_KERNEL_INITIALIZER,
                    kernel_regularizer=l2(weight_decay), name=name + 'se_expand')(se)
        x = multiply([x, se], name=name + 'se_excite')

    # part3 1×1逐点卷积恢复通道维数
    x = Conv2D(filters_out, 1, 1, 'same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay), name=name + 'project_conv')(x)
    x = BatchNormalization(axis=-1, name=name + 'project_bn')(x)

    # drop connect and skip connection
    if id_skip is True and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = Dropout(rate=drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)

        x = add([x, inputs], name=name + 'add')

    return x


def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(tf.math.ceil(depth_coefficient * repeats))


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation=tf.nn.swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 weight_decay=1e-4,
                 **kwargs):
    if input_tensor is not None:
        img_input = input_tensor
    else:
        if input_shape is None:
            img_input = Input(shape=[default_size, default_size, 3])
        else:
            img_input = Input(shape=input_shape)

    # Build stem
    x = img_input
    x = Conv2D(round_filters(32, width_coefficient, depth_divisor), 3, 2, 'same', use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER, kernel_regularizer=l2(weight_decay), name='stem_conv')(x)
    x = BatchNormalization(name='stem_bn')(x)
    x = Activation(activation, name='stem_activation')(x)

    # Build blocks
    blocks_args = deepcopy(blocks_args)
    block_num = 0
    # 计算总的block的数量
    num_blocks_total = float(sum(block_args['repeats'] for block_args in blocks_args))
    for index, block_args in enumerate(blocks_args):
        assert block_args['repeats'] > 0
        block_args['filters_in'] = round_filters(block_args['filters_in'], width_coefficient, depth_divisor)
        block_args['filters_out'] = round_filters(block_args['filters_out'], width_coefficient, depth_divisor)
        block_args['repeats'] = round_repeats(block_args['repeats'], depth_coefficient)
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total

        for r in range(block_args.pop('repeats')):
            if r > 0:
                block_args['strides'] = 1
                block_args['filters_in'] = block_args['filters_out']
            x = block(x, activation, drop_rate, name='block{}{}_'.format(index + 1, chr(r + 97)), **block_args)
            block_num += 1

    # Build top
    x = Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1, 1, 'same', use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER, kernel_regularizer=l2(weight_decay), name='top_conv')(x)
    x = BatchNormalization(axis=-1, name='top_bn')(x)
    x = Activation(activation, name='top_activation')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = Dropout(dropout_rate, name='top_dropout')(x)
        x = Dense(classes, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER,
                  kernel_regularizer=l2(weight_decay), name='probs')(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    inputs = img_input

    # Create model
    model = tf.keras.models.Model(inputs, x, name=model_name)

    # Load weights
    if weights == 'imagenet':
        if include_top:
            file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
            file_hash = IMAGENET_WEIGHTS_HASHES[model_name][0]
        else:
            file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
            file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
        file_name = model_name + file_suff
        weights_path = tf.keras.utils.get_file(file_name,
                                               BASE_WEIGHTS_PATH + file_name,
                                               cache_subdir='models',
                                               file_hash=file_hash)

        model.load_weights(weights_path)

    elif weights == 'noisy-student':

        if include_top:
            file_name = "{}_{}.h5".format(model_name, weights)
            file_hash = NS_WEIGHTS_HASHES[model_name][0]
        else:
            file_name = "{}_{}_notop.h5".format(model_name, weights)
            file_hash = NS_WEIGHTS_HASHES[model_name][1]
        weights_path = tf.keras.utils.get_file(file_name,
                                               NS_WEIGHTS_PATH + file_name,
                                               cache_subdir='models',
                                               file_hash=file_hash)

        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


def EfficientNetB0(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):

    return EfficientNet(1.0, 1.0, 224, 0.2,
                        model_name='efficientnet-b0',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB1(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):

    return EfficientNet(1.0, 1.1, 240, 0.2,
                        model_name='efficientnet-b1',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):

    return EfficientNet(1.1, 1.2, 260, 0.3,
                        model_name='efficientnet-b2',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):

    return EfficientNet(1.2, 1.4, 300, 0.3,
                        model_name='efficientnet-b3',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB4(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):

    return EfficientNet(1.4, 1.8, 380, 0.4,
                        model_name='efficientnet-b4',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB5(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):

    return EfficientNet(1.6, 2.2, 456, 0.4,
                        model_name='efficientnet-b5',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB6(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):

    return EfficientNet(1.8, 2.6, 528, 0.5,
                        model_name='efficientnet-b6',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB7(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):

    return EfficientNet(2.0, 3.1, 600, 0.5,
                        model_name='efficientnet-b7',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetL2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):

    return EfficientNet(4.3, 5.3, 800, 0.5,
                        model_name='efficientnet-l2',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


if __name__ == '__main__':
    model1 = EfficientNetB6()
    model1.summary()