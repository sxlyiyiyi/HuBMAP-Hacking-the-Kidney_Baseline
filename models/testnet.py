from functools import wraps
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Conv2DTranspose, UpSampling2D, Conv2D, Add
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate, ReLU, Multiply
from tensorflow.keras.models import Model

from models.backbone.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.backbone.resnext import ResNeXt50, ResNeXt101
from models.backbone.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from models.backbone.efficientnet import EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetL2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from models.backbone.senet import SENet154, SEResNet50, SEResNet101, SEResNet152, SEResNeXt50, SEResNeXt101


DEFAULT_SKIP_CONNECTIONS = {
    'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),
    'resnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),  # check 'bn_data'
    'resnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet152': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnext101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'inceptionv3': (228, 86, 16, 9),
    'inceptionresnetv2': (594, 260, 16, 9),
    'densenet121': (311, 139, 51, 4),
    'densenet169': (367, 139, 51, 4),
    'densenet201': (479, 139, 51, 4),
    'efficientnetb0': ('block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb1': ('block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb2': ('block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb3': ('block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb4': ('block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb5': ('block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb6': ('block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb7': ('block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetl2': ('block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation', 'block2a_expand_activation'),
    'seresnext50': ('activation_65', 'activation_35', 'activation_15', 'activation'),
    'seresnext101': ('activation_150', 'activation_35', 'activation_15', 'activation'),
    'seresnet50': ('activation_65', 'activation_35', 'activation_15', 'activation'),
    'seresnet101': ('activation_150', 'activation_35', 'activation_15', 'activation'),
    'seresnet152': ('activation_235', 'activation_55', 'activation_15', 'activation'),
    'senet154': ('activation_237', 'activation_55', 'activation_17', 'activation_2'),
}


def UUnet(backbone_name='vgg16',
         input_shape=(None, None, 3),
         input_tensor=None,
         encoder_weights='imagenet',
         freeze_encoder=False,
         skip_connections='default',
         decoder_block_type='upsampling',
         decoder_filters=(256, 128, 64, 32, 16),
         decoder_use_batchnorm=True,
         n_upsample_blocks=5,
         upsample_rates=(2, 2, 2, 2, 2),
         classes=1,
         activation='sigmoid'):
    """

    Args:
        backbone_name: (str) look at list of available backbones.
        input_shape:  (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            'dof' (pre-training on DoF)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        skip_connections: if 'default' is used take default skip connections,
            else provide a list of layer numbers or names starting from top of model
        decoder_block_type: (str) one of 'upsampling' and 'transpose' (look at blocks.py)
        decoder_filters: (int) number of convolution layer filters in decoder blocks
        decoder_use_batchnorm: (bool) if True add batch normalisation layer between `Conv2D` ad `Activation` layers
        n_upsample_blocks: (int) a number of upsampling blocks
        upsample_rates: (tuple of int) upsampling rates decoder blocks
        classes: (int) a number of classes for output
        activation: (str) one of keras activations for last model layer

    Returns:
        keras.models.Model instance

    """

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=input_tensor,
                            weights=encoder_weights,
                            include_top=False)

    if skip_connections == 'default':
        skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name]

    model = build_unet(backbone,
                       classes,
                       skip_connections,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=n_upsample_blocks,
                       upsample_rates=upsample_rates,
                       use_batchnorm=decoder_use_batchnorm)

    # lock encoder weights for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    model._name = 'u-{}'.format(backbone_name)

    return model


def build_unet(backbone, classes, skip_connection_layers,
               decoder_filters=(256, 128, 64, 32, 16),
               upsample_rates=(2, 2, 2, 2, 2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True):
    inputs = backbone.input
    x = backbone.output

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])

    d3 = backbone.layers[skip_connection_idx[2]].output
    d4 = backbone.layers[skip_connection_idx[1]].output
    d5 = backbone.layers[skip_connection_idx[0]].output

    rfb_5 = rfb(d5, 32)
    rfb_4 = rfb(d4, 32)
    rfb_3 = rfb(d3, 32)

    ra5_feat = aggregation(rfb_5, rfb_4, rfb_3, 32)
    feat_map5 = tf.image.resize(ra5_feat, size=(tf.shape(inputs)[1], tf.shape(inputs)[2]))

    crop_4 = tf.image.resize(ra5_feat, size=(tf.shape(d5)[1], tf.shape(d5)[2]))
    x = -1 * tf.sigmoid(crop_4) + 1
    x = Multiply()([x, d5])
    x = conv2d_bn_relu(x, 256, 1)
    x = conv2d_bn_relu(x, 256, 3)
    x = conv2d_bn_relu(x, 256, 3)
    x = conv2d_bn_relu(x, 256, 3)
    ra4_feat = Conv2D(1, 1, padding='same')(x)
    x = ra4_feat + crop_4
    feat_map4 = tf.image.resize(x, size=(tf.shape(inputs)[1], tf.shape(inputs)[2]))

    crop_3 = tf.image.resize(x, size=(tf.shape(d4)[1], tf.shape(d4)[2]))
    x = -1 * tf.sigmoid(crop_3) + 1
    x = Multiply()([x, d4])
    x = conv2d_bn_relu(x, 64, 1)
    x = conv2d_bn_relu(x, 64, 3)
    ra3_feat = Conv2D(1, 1, padding='same')(x)
    x = ra3_feat + crop_3
    feat_map3 = tf.image.resize(x, size=(tf.shape(inputs)[1], tf.shape(inputs)[2]))

    crop_2 = tf.image.resize(x, size=(tf.shape(d4)[1], tf.shape(d4)[2]))
    x = -1 * tf.sigmoid(crop_2) + 1
    x = Multiply()([x, d3])
    x = conv2d_bn_relu(x, 64, 1)
    x = conv2d_bn_relu(x, 64, 3)
    ra2_feat = Conv2D(1, 1, padding='same')(x)
    x = ra2_feat + crop_2
    feat_map2 = tf.image.resize(x, size=(tf.shape(inputs)[1], tf.shape(inputs)[2]))

    if activation in {'softmax', 'sigmoid'}:
        feat_map2 = Activation(activation)(feat_map2)
        feat_map3 = Activation(activation)(feat_map3)
        feat_map4 = Activation(activation)(feat_map4)
        feat_map5 = Activation(activation)(feat_map5)

    return Model(inputs, (feat_map5, feat_map4, feat_map3, feat_map2))


def aggregation(x1, x2, x3, channel):
    x1_1 = x1
    x2_1 = UpSampling2D(2, interpolation='bilinear')(x1)
    x2_1 = conv2d_bn_relu(x2_1, channel, 3) * x2
    x3_1_1 = UpSampling2D(4, interpolation='bilinear')(x1)
    x3_1_1 = conv2d_bn_relu(x3_1_1, channel, 3)
    x3_1_2 = UpSampling2D(2, interpolation='bilinear')(x2)
    x3_1 = x3_1_1 * conv2d_bn_relu(x3_1_2, channel, 3) * x3

    x2_2_1 = UpSampling2D(2, interpolation='bilinear')(x1_1)
    x2_2_1 = conv2d_bn_relu(x2_2_1, channel, 3)
    x2_2 = Concatenate()([x2_1, x2_2_1])
    x2_2 = conv2d_bn_relu(x2_2, 2 * channel, 3)

    x3_2_2 = UpSampling2D(2, interpolation='bilinear')(x2_2)
    x3_2_2 = conv2d_bn_relu(x3_2_2, channel * 2, 3)
    x3_2 = Concatenate()([x3_1, x3_2_2])
    x3_2 = conv2d_bn_relu(x3_2, channel * 3, 3)

    x = conv2d_bn_relu(x3_2, 3 * channel, 1)
    x = Conv2D(1, 1, 1, padding='same')(x)

    return x


def rfb(x, channel):
    x0 = conv2d_bn_relu(x, channel, 1)

    x1 = conv2d_bn_relu(x, channel, 1)
    x1 = conv2d_bn_relu(x1, channel, (1, 3))
    x1 = conv2d_bn_relu(x1, channel, (3, 1))
    x1 = conv2d_bn_relu(x1, channel, 3, dilation_rate=3)

    x2 = conv2d_bn_relu(x, channel, 1)
    x2 = conv2d_bn_relu(x2, channel, (1, 5))
    x2 = conv2d_bn_relu(x2, channel, (5, 1))
    x2 = conv2d_bn_relu(x2, channel, 3, dilation_rate=5)

    x3 = conv2d_bn_relu(x, channel, 1)
    x3 = conv2d_bn_relu(x3, channel, (1, 7))
    x3 = conv2d_bn_relu(x3, channel, (7, 1))
    x3 = conv2d_bn_relu(x3, channel, 3, dilation_rate=7)

    x_cat = Concatenate()([x0, x1, x2, x3])
    x_cat = conv2d_bn_relu(x_cat, channel, 3)

    x_res = conv2d_bn_relu(x, channel, 1)
    x = Add()([x_cat, x_res])
    x = ReLU()(x)

    return x


def handle_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def conv2d_bn_relu(inputs, channels, kernel_size, stride=1, padding='same', dilation_rate=(1, 1)):
    x = Conv2D(channels, kernel_size, stride, padding, use_bias=False, dilation_rate=dilation_rate)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not use_batchnorm)(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x

    return layer


def Upsample2D_block(filters, stage, kernel_size=(3, 3), upsample_rate=(2, 2),
                     use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x

    return layer


def Transpose2D_block(filters, stage, kernel_size=(3, 3), upsample_rate=(2, 2),
                      transpose_kernel_size=(4, 4), use_batchnorm=False, skip=None):
    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not (use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name + '1')(x)
        x = Activation('relu', name=relu_name + '1')(x)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x

    return layer


def get_layer_number(model, layer_name):
    """
    Help find layer in Keras model by name
    Args:
        model: Keras `Model`
        layer_name: str, name of layer

    Returns:
        index of layer

    Raises:
        ValueError: if model does not contains layer with such name
    """
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))


def extract_outputs(model, layers, include_top=False):
    """
    Help extract intermediate layer outputs from model
    Args:
        model: Keras `Model`
        layer: list of integers/str, list of layers indexes or names to extract output
        include_top: bool, include final model layer output

    Returns:
        list of tensors (outputs)
    """
    layers_indexes = ([get_layer_number(model, l) if isinstance(l, str) else l
                       for l in layers])
    outputs = [model.layers[i].output for i in layers_indexes]

    if include_top:
        outputs.insert(0, model.output)

    return outputs


def reverse(l):
    """Reverse list"""
    return list(reversed(l))


# decorator for models aliases, to add doc string
def add_docstring(doc_string=None):
    def decorator(fn):
        if fn.__doc__:
            fn.__doc__ += doc_string
        else:
            fn.__doc__ = doc_string

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def recompile(model):
    model.compile(model.optimizer, model.loss, model.metrics)


def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False
    return


def set_trainable(model):
    for layer in model.layers:
        layer.trainable = True
    recompile(model)


def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))


backbones = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "resnext50": ResNeXt50,
    "resnext101": ResNeXt101,
    "inceptionresnetv2": InceptionResNetV2,
    "inceptionv3": InceptionV3,
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
    "efficientnetb0": EfficientNetB0,
    "efficientnetb1": EfficientNetB1,
    "efficientnetb2": EfficientNetB2,
    "efficientnetb3": EfficientNetB3,
    "efficientnetb4": EfficientNetB4,
    "efficientnetb5": EfficientNetB5,
    "efficientnetb6": EfficientNetB6,
    "efficientnetb7": EfficientNetB7,
    "efficientnetl2": EfficientNetL2,
    "seresnext50": SEResNeXt50,
    "seresnext101": SEResNeXt101,
    "seresnet50": SEResNet50,
    "seresnet101": SEResNet101,
    "seresnet152": SEResNet152,
    'senet154': SENet154
}


def get_backbone(name, *args, **kwargs):
    return backbones[name](*args, **kwargs)


if __name__ == "__main__":
    model1 = UUnet('efficientnetb2', (352, 352, 3), encoder_weights='imagenet')
    model1.summary()
