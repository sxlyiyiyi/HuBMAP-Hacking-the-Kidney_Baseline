import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Lambda
from tensorflow.keras.layers import Concatenate, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras.utils import get_file
from keras_applications.imagenet_utils import _obtain_input_shape

weights_collection = [
# ResNeXt50
    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext50_imagenet_1000.h5',
        'name': 'resnext50_imagenet_1000.h5',
        'md5': '7c5c40381efb044a8dea5287ab2c83db',
    },

    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext50_imagenet_1000_no_top.h5',
        'name': 'resnext50_imagenet_1000_no_top.h5',
        'md5': '7ade5c8aac9194af79b1724229bdaa50',
    },


    # ResNeXt101
    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext101_imagenet_1000.h5',
        'name': 'resnext101_imagenet_1000.h5',
        'md5': '432536e85ee811568a0851c328182735',
    },

    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext101_imagenet_1000_no_top.h5',
        'name': 'resnext101_imagenet_1000_no_top.h5',
        'md5': '91fe0126320e49f6ee607a0719828c7e',
    },
]


def ResNeXt50(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = build_resnext(input_tensor=input_tensor,
                          input_shape=input_shape,
                          first_block_filters=128,
                          repetitions=(3, 4, 6, 3),
                          classes=classes,
                          include_top=include_top)
    model._name = 'resnext50'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def ResNeXt101(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = build_resnext(input_tensor=input_tensor,
                          input_shape=input_shape,
                          first_block_filters=128,
                          repetitions=(3, 4, 23, 3),
                          classes=classes,
                          include_top=include_top)
    model._name = 'resnext101'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def build_resnext(
        repetitions=(2, 2, 2, 2),
        include_top=True,
        input_tensor=None,
        input_shape=None,
        classes=1000,
        first_conv_filters=64,
        first_block_filters=64):
    """
    TODO
    """

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format='channels_last',
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = first_block_filters

    # resnext bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(first_conv_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = BatchNormalization(name='bn0', **bn_params)(x)
    x = Activation('relu', name='relu0')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnext body
    for stage, rep in enumerate(repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if stage == 0 and block == 0:
                x = conv_block(filters, stage, block, strides=(1, 1))(x)

            elif block == 0:
                x = conv_block(filters, stage, block, strides=(2, 2))(x)

            else:
                x = identity_block(filters, stage, block)(x)

    # resnext top
    if include_top:
        x = GlobalAveragePooling2D(name='pool1')(x)
        x = Dense(classes, name='fc1')(x)
        x = Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = Model(inputs, x)

    return model


def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def GroupConv2D(filters, kernel_size, conv_params, conv_name, strides=(1,1), cardinality=32):

    def layer(input_tensor):

        grouped_channels = int(input_tensor.shape[-1]) // cardinality

        blocks = []
        for c in range(cardinality):
            x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input_tensor)
            name = conv_name + '_' + str(c)
            x = Conv2D(grouped_channels, kernel_size, strides=strides,
                       name=name, **conv_params)(x)
            blocks.append(x)

        x = Concatenate(axis=-1)(blocks)
        return x
    return layer


def conv_block(filters, stage, block, strides=(2, 2)):
    """The conv block is the block that has conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        strides: tuple of integers, strides for conv (3x3) layer in block
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):

        # extracting params and names for layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), conv_params, conv_name + '2', strides=strides)(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)

        x = Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)

        shortcut = Conv2D(filters*2, (1, 1), name=sc_name, strides=strides, **conv_params)(input_tensor)
        shortcut = BatchNormalization(name=sc_name+'_bn', **bn_params)(shortcut)
        x = Add()([x, shortcut])

        x = Activation('relu', name=relu_name)(x)
        return x

    return layer


def identity_block(filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), conv_params, conv_name + '2')(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)

        x = Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)

        x = Add()([x, input_tensor])

        x = Activation('relu', name=relu_name)(x)
        return x

    return layer


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'glorot_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


def find_weights(_weights_collection, model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, _weights_collection))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    return w


def load_model_weights(_weights_collection, model, dataset, classes, include_top):
    weights = find_weights(_weights_collection, model.name, dataset, include_top)

    if weights:
        weights = weights[0]

        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))

        weights_path = get_file(weights['name'],
                                weights['url'],
                                cache_subdir='models',
                                md5_hash=weights['md5'])

        model.load_weights(weights_path)

    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))


if __name__ == '__main__':
    from tensorflow.keras.utils import plot_model
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

    model1 = ResNeXt50(input_shape=(512, 512, 3), weights='imagenet', include_top=False)
    model1.summary()
    plot_model(model1, to_file='./ResNext.png', show_shapes=True,)
