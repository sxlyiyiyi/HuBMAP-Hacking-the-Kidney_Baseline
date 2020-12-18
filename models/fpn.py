from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models

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


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name
        )(input_tensor)

    return wrapper


def DoubleConv3x3BnReLU(filters, use_batchnorm, name=None):
    name1, name2 = None, None
    if name is not None:
        name1 = name + 'a'
        name2 = name + 'b'

    def wrapper(input_tensor):
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name1)(input_tensor)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name2)(x)
        return x

    return wrapper


def FPNBlock(pyramid_filters, stage):
    conv0_name = 'fpn_stage_p{}_pre_conv'.format(stage)
    conv1_name = 'fpn_stage_p{}_conv'.format(stage)
    add_name = 'fpn_stage_p{}_add'.format(stage)
    up_name = 'fpn_stage_p{}_upsampling'.format(stage)

    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip):
        # if input tensor channels not equal to pyramid channels
        # we will not be able to sum input tensor and skip
        # so add extra conv layer to transform it
        input_filters = backend.int_shape(input_tensor)[channels_axis]
        if input_filters != pyramid_filters:
            input_tensor = layers.Conv2D(
                filters=pyramid_filters,
                kernel_size=(1, 1),
                kernel_initializer='he_uniform',
                name=conv0_name,
            )(input_tensor)

        skip = layers.Conv2D(
            filters=pyramid_filters,
            kernel_size=(1, 1),
            kernel_initializer='he_uniform',
            name=conv1_name,
        )(skip)

        x = layers.UpSampling2D((2, 2), name=up_name)(input_tensor)
        x = layers.Add(name=add_name)([x, skip])

        return x

    return wrapper


# ---------------------------------------------------------------------
#  FPN Decoder
# ---------------------------------------------------------------------

def build_fpn(
        backbone,
        skip_connection_layers,
        pyramid_filters=256,
        segmentation_filters=128,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        aggregation='sum',
        dropout=None,
):
    input_ = backbone.input
    x = backbone.output

    # building decoder blocks with skip connections
    ls = ([get_layer_number(backbone, l) if isinstance(l, str) else l for l in skip_connection_layers])
    skips = ([backbone.layers[c].output for c in ls])
    # x = (backbone.get_layer(name=psp_layer_idx).output if isinstance(psp_layer_idx, str)
    #      else backbone.get_layer(index=psp_layer_idx).output)

    # build FPN pyramid
    p5 = FPNBlock(pyramid_filters, stage=5)(x, skips[0])
    p4 = FPNBlock(pyramid_filters, stage=4)(p5, skips[1])
    p3 = FPNBlock(pyramid_filters, stage=3)(p4, skips[2])
    p2 = FPNBlock(pyramid_filters, stage=2)(p3, skips[3])

    # add segmentation head to each
    s5 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage2')(p2)

    # upsampling to same resolution
    s5 = layers.UpSampling2D((8, 8), interpolation='nearest', name='upsampling_stage5')(s5)
    s4 = layers.UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage4')(s4)
    s3 = layers.UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage3')(s3)

    # aggregating results
    if aggregation == 'sum':
        x = layers.Add(name='aggregation_sum')([s2, s3, s4, s5])
    elif aggregation == 'concat':
        concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x = layers.Concatenate(axis=concat_axis, name='aggregation_concat')([s2, s3, s4, s5])
    else:
        raise ValueError('Aggregation parameter should be in ("sum", "concat"), '
                         'got {}'.format(aggregation))

    if dropout:
        x = layers.SpatialDropout2D(dropout, name='pyramid_dropout')(x)

    # final stage
    x = Conv3x3BnReLU(segmentation_filters, use_batchnorm, name='final_stage')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='final_upsampling')(x)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv',
    )(x)

    if activation in {'softmax', 'sigmoid'}:
        x = layers.Activation(activation, name=activation)(x)
    # x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  FPN Model
# ---------------------------------------------------------------------

def FPN(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=21,
        activation='softmax',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        pyramid_block_filters=256,
        pyramid_use_batchnorm=True,
        pyramid_aggregation='concat',
        pyramid_dropout=None,
        **kwargs
):
    """FPN_ is a fully convolution neural network for image semantic segmentation
    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
                case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
                able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        weights: optional, path to model weights.
        activation: name of one of ``keras.activations`` for last model layer (e.g. ``sigmoid``, ``softmax``, ``linear``).
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
                Each of these layers will be used to build features pyramid. If ``default`` is used
                layer names are taken from ``DEFAULT_FEATURE_PYRAMID_LAYERS``.
        pyramid_block_filters: a number of filters in Feature Pyramid Block of FPN_.
        pyramid_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.
        pyramid_aggregation: one of 'sum' or 'concat'. The way to aggregate pyramid blocks.
        pyramid_dropout: spatial dropout rate for feature pyramid in range (0, 1).
    Returns:
        ``keras.models.Model``: **FPN**
    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
    """
    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            weights=encoder_weights,
                            include_top=False,
                            **kwargs)

    if encoder_features == 'default':
        encoder_features = DEFAULT_SKIP_CONNECTIONS[backbone_name]

    model = build_fpn(
        backbone=backbone,
        skip_connection_layers=encoder_features,
        pyramid_filters=pyramid_block_filters,
        segmentation_filters=pyramid_block_filters // 2,
        use_batchnorm=pyramid_use_batchnorm,
        dropout=pyramid_dropout,
        activation=activation,
        classes=classes,
        aggregation=pyramid_aggregation,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model


def freeze_model(model, **kwargs):
    """Set all layers non trainable, excluding BatchNormalization layers"""
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return


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
    from tensorflow.keras.utils import plot_model
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

    model1 = FPN('seresnext50', (512, 512, 3), classes=2)
    model1.summary()
    plot_model(model1, to_file='./seg_hrnet.png', show_shapes=True)