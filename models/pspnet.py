import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.keras.utils as keras_utils

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
    'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2'),
    'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2'),
    'resnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
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


# ---------------------------------------------------------------------
#  PSP Model
# ---------------------------------------------------------------------

def PSPNet(
        backbone_name='vgg16',
        input_shape=(384, 384, 3),
        classes=21,
        activation='softmax',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        downsample_factor=8,
        psp_conv_filters=512,
        psp_pooling_type='avg',
        psp_use_batchnorm=True,
        psp_dropout=None,
        **kwargs
):
    """PSPNet_ is a fully convolution neural network for image semantic segmentation
    Args:
        backbone_name: name of classification model used as feature
                extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``.
            ``H`` and ``W`` should be divisible by ``6 * downsample_factor`` and **NOT** ``None``!
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
                (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        downsample_factor: one of 4, 8 and 16. Downsampling rate or in other words backbone depth
            to construct PSP module on it.
        psp_conv_filters: number of filters in ``Conv2D`` layer in each PSP block.
        psp_pooling_type: one of 'avg', 'max'. PSP block pooling type (maximum or average).
        psp_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.
        psp_dropout: dropout rate between 0 and 1.
    Returns:
        ``keras.models.Model``: **PSPNet**
    .. _PSPNet:
        https://arxiv.org/pdf/1612.01105.pdf
    """
    # control image input shape
    check_input_shape(input_shape, downsample_factor)

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            weights=encoder_weights,
                            include_top=False)

    feature_layers = DEFAULT_SKIP_CONNECTIONS[backbone_name]

    if downsample_factor == 16:
        psp_layer_idx = feature_layers[0]
    elif downsample_factor == 8:
        psp_layer_idx = feature_layers[1]
    elif downsample_factor == 4:
        psp_layer_idx = feature_layers[2]
    else:
        raise ValueError('Unsupported factor - `{}`, Use 4, 8 or 16.'.format(downsample_factor))

    model = build_psp(
        backbone,
        psp_layer_idx,
        pooling_type=psp_pooling_type,
        conv_filters=psp_conv_filters,
        use_batchnorm=psp_use_batchnorm,
        final_upsampling_factor=downsample_factor,
        classes=classes,
        activation=activation,
        dropout=psp_dropout,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model


# ---------------------------------------------------------------------
#  PSP Decoder
# ---------------------------------------------------------------------

def build_psp(
        backbone,
        psp_layer_idx,
        pooling_type='avg',
        conv_filters=512,
        use_batchnorm=True,
        final_upsampling_factor=8,
        classes=21,
        activation='softmax',
        dropout=None,
):
    input_ = backbone.input
    x = (backbone.get_layer(name=psp_layer_idx).output if isinstance(psp_layer_idx, str)
         else backbone.get_layer(index=psp_layer_idx).output)
    # x = (get_layer_number(backbone, psp_layer_idx) if isinstance(psp_layer_idx, str) else psp_layer_idx)

    # build spatial pyramid
    x1 = SpatialContextBlock(1, conv_filters, pooling_type, use_batchnorm)(x)
    x2 = SpatialContextBlock(2, conv_filters, pooling_type, use_batchnorm)(x)
    x3 = SpatialContextBlock(3, conv_filters, pooling_type, use_batchnorm)(x)
    x6 = SpatialContextBlock(6, conv_filters, pooling_type, use_batchnorm)(x)

    # aggregate spatial pyramid
    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.Concatenate(axis=concat_axis, name='psp_concat')([x, x1, x2, x3, x6])
    x = Conv1x1BnReLU(conv_filters, use_batchnorm, name='aggregation')(x)

    # model regularization
    if dropout is not None:
        x = layers.SpatialDropout2D(dropout, name='spatial_dropout')(x)

    # model head
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)

    x = layers.UpSampling2D(final_upsampling_factor, name='final_upsampling', interpolation='bilinear')(x)
    if activation in {'softmax', 'sigmoid'}:
        x = layers.Activation(activation, name=activation)(x)

    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------
def Conv2dBn(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs
):
    """Extension of Conv2D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)

    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor):

        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not use_batchnorm,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper


def check_input_shape(input_shape, factor):
    if input_shape is None:
        raise ValueError("Input shape should be a tuple of 3 integers, not None!")

    h, w = input_shape[:2] if backend.image_data_format() == 'channels_last' else input_shape[1:]
    min_size = factor * 6

    is_wrong_shape = (
            h % min_size != 0 or w % min_size != 0 or
            h < min_size or w < min_size
    )

    if is_wrong_shape:
        raise ValueError('Wrong shape {}, input H and W should '.format(input_shape) +
                         'be divisible by `{}`'.format(min_size))


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv1x1BnReLU(filters, use_batchnorm, name=None):

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name
        )(input_tensor)

    return wrapper


def SpatialContextBlock(
        level,
        conv_filters=512,
        pooling_type='avg',
        use_batchnorm=True,
):
    if pooling_type not in ('max', 'avg'):
        raise ValueError('Unsupported pooling type - `{}`.'.format(pooling_type) +
                         'Use `avg` or `max`.')

    Pooling2D = layers.MaxPool2D if pooling_type == 'max' else layers.AveragePooling2D

    pooling_name = 'psp_level{}_pooling'.format(level)
    conv_block_name = 'psp_level{}'.format(level)
    upsampling_name = 'psp_level{}_upsampling'.format(level)

    def wrapper(input_tensor):
        # extract input feature maps size (h, and w dimensions)
        input_shape = backend.int_shape(input_tensor)
        spatial_size = input_shape[1:3] if backend.image_data_format() == 'channels_last' else input_shape[2:]

        # Compute the kernel and stride sizes according to how large the final feature map will be
        # When the kernel factor and strides are equal, then we can compute the final feature map factor
        # by simply dividing the current factor by the kernel or stride factor
        # The final feature map sizes are 1x1, 2x2, 3x3, and 6x6.
        pool_size = up_size = [spatial_size[0] // level, spatial_size[1] // level]

        x = Pooling2D(pool_size, strides=pool_size, padding='same', name=pooling_name)(input_tensor)
        x = Conv1x1BnReLU(conv_filters, use_batchnorm, name=conv_block_name)(x)
        x = layers.UpSampling2D(up_size, interpolation='bilinear', name=upsampling_name)(x)
        return x

    return wrapper


def freeze_model(model, **kwargs):
    """Set all layers non trainable, excluding BatchNormalization layers"""
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return


def filter_keras_submodules(kwargs):
    """Selects only arguments that define keras_application submodules. """
    submodule_keys = kwargs.keys() & {'backend', 'layers', 'models', 'utils'}
    return {key: kwargs[key] for key in submodule_keys}


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
    model1 = PSPNet('efficientnetb4', (1200, 1200, 3), encoder_weights='imagenet')
    model1.summary()