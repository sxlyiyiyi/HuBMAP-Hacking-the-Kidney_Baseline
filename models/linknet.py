import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose as Transpose
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


def Linknet(backbone_name='vgg16',
            input_shape=(None, None, 3),
            input_tensor=None,
            encoder_weights='imagenet',
            freeze_encoder=False,
            skip_connections='default',
            n_upsample_blocks=5,
            decoder_filters=(None, None, None, None, 16),
            decoder_use_batchnorm=True,
            upsample_layer='upsampling',
            upsample_kernel_size=(3, 3),
            classes=1,
            activation='sigmoid'):
    """
    Version of Linkent model (https://arxiv.org/pdf/1707.03718.pdf)
    This implementation by default has 4 skip connection links (original - 3).

    Args:
        backbone_name: (str) look at list of available backbones.
        input_shape: (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        skip_connections: if 'default' is used take default skip connections,
        decoder_filters: (tuple of int) a number of convolution filters in decoder blocks,
            for block with skip connection a number of filters is equal to number of filters in
            corresponding encoder block (estimates automatically and can be passed as `None` value).
        decoder_use_batchnorm: (bool) if True add batch normalisation layer between `Conv2D` ad `Activation` layers
        n_upsample_blocks: (int) a number of upsampling blocks in decoder
        upsample_layer: (str) one of 'upsampling' and 'transpose'
        upsample_kernel_size: (tuple of int) convolution kernel size in upsampling block
        classes: (int) a number of classes for output
        activation: (str) one of keras activations

    Returns:
        model: instance of Keras Model

    """

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=input_tensor,
                            weights=encoder_weights,
                            include_top=False)

    if skip_connections == 'default':
        skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name]

    model = build_linknet(backbone,
                          classes,
                          skip_connections,
                          decoder_filters=decoder_filters,
                          upsample_layer=upsample_layer,
                          activation=activation,
                          n_upsample_blocks=n_upsample_blocks,
                          upsample_rates=(2, 2, 2, 2, 2),
                          upsample_kernel_size=upsample_kernel_size,
                          use_batchnorm=decoder_use_batchnorm)

    # lock encoder weights for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    model._name = 'link-{}'.format(backbone_name)

    return model


def build_linknet(backbone,
                  classes,
                  skip_connection_layers,
                  decoder_filters=(None, None, None, None, 16),
                  upsample_rates=(2, 2, 2, 2, 2),
                  n_upsample_blocks=5,
                  upsample_kernel_size=(3, 3),
                  upsample_layer='upsampling',
                  activation='sigmoid',
                  use_batchnorm=True):
    inputs = backbone.input
    x = backbone.output

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = DecoderBlock(stage=i,
                         filters=decoder_filters[i],
                         kernel_size=upsample_kernel_size,
                         upsample_rate=upsample_rate,
                         use_batchnorm=use_batchnorm,
                         upsample_layer=upsample_layer,
                         skip=skip_connection)(x)

    x = Conv2D(classes, (3, 3), padding='same', name='final_conv')(x)

    if activation in {'softmax', 'sigmoid'}:
        x = Activation(activation, name=activation)(x)

    model = Model(inputs, x)

    return model


def handle_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def ConvRelu(filters,
             kernel_size,
             use_batchnorm=False,
             conv_name='conv',
             bn_name='bn',
             relu_name='relu'):
    def layer(x):
        x = Conv2D(filters,
                   kernel_size,
                   padding="same",
                   name=conv_name,
                   use_bias=not (use_batchnorm))(x)

        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)

        x = Activation('relu', name=relu_name)(x)

        return x

    return layer


def Conv2DUpsample(filters,
                   upsample_rate,
                   kernel_size=(3, 3),
                   up_name='up',
                   conv_name='conv',
                   **kwargs):
    def layer(input_tensor):
        x = UpSampling2D(upsample_rate, name=up_name)(input_tensor)
        x = Conv2D(filters,
                   kernel_size,
                   padding='same',
                   name=conv_name,
                   **kwargs)(x)
        return x

    return layer


def Conv2DTranspose(filters,
                    upsample_rate,
                    kernel_size=(4, 4),
                    up_name='up',
                    **kwargs):
    if not tuple(upsample_rate) == (2, 2):
        raise NotImplementedError(
            f'Conv2DTranspose support only upsample_rate=(2, 2), got {upsample_rate}')

    def layer(input_tensor):
        x = Transpose(filters,
                      kernel_size=kernel_size,
                      strides=upsample_rate,
                      padding='same',
                      name=up_name)(input_tensor)
        return x

    return layer


def UpsampleBlock(filters,
                  upsample_rate,
                  kernel_size,
                  use_batchnorm=False,
                  upsample_layer='upsampling',
                  conv_name='conv',
                  bn_name='bn',
                  relu_name='relu',
                  up_name='up',
                  **kwargs):
    if upsample_layer == 'upsampling':
        UpBlock = Conv2DUpsample

    elif upsample_layer == 'transpose':
        UpBlock = Conv2DTranspose

    else:
        raise ValueError(f'Not supported up layer type {upsample_layer}')

    def layer(input_tensor):

        x = UpBlock(filters,
                    upsample_rate=upsample_rate,
                    kernel_size=kernel_size,
                    use_bias=not use_batchnorm,
                    conv_name=conv_name,
                    up_name=up_name,
                    **kwargs)(input_tensor)

        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)

        x = Activation('relu', name=relu_name)(x)

        return x

    return layer


def DecoderBlock(stage,
                 filters=None,
                 kernel_size=(3, 3),
                 upsample_rate=(2, 2),
                 use_batchnorm=False,
                 skip=None,
                 upsample_layer='upsampling'):
    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)
        input_filters = K.int_shape(input_tensor)[-1]

        if skip is not None:
            output_filters = K.int_shape(skip)[-1]
        else:
            output_filters = filters

        x = ConvRelu(input_filters // 4,
                     kernel_size=(1, 1),
                     use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1',
                     bn_name=bn_name + '1',
                     relu_name=relu_name + '1')(input_tensor)

        x = UpsampleBlock(filters=input_filters // 4,
                          kernel_size=kernel_size,
                          upsample_layer=upsample_layer,
                          upsample_rate=upsample_rate,
                          use_batchnorm=use_batchnorm,
                          conv_name=conv_name + '2',
                          bn_name=bn_name + '2',
                          up_name=up_name + '2',
                          relu_name=relu_name + '2')(x)

        x = ConvRelu(output_filters,
                     kernel_size=(1, 1),
                     use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '3',
                     bn_name=bn_name + '3',
                     relu_name=relu_name + '3')(x)

        if skip is not None:
            x = Add()([x, skip])

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


def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))


def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False
    return


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


if __name__ == '__main__':
    from tensorflow.keras.utils import plot_model
    import os

    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'
    # model = SENet154(input_shape=(512, 512, 3), include_top=False)
    # for layer in model.layers:
    #    print(layer.output)
    # model.summary()

    # model2 = EfficientNetB0(input_shape=(512, 512, 3), include_top=False)
    # model2.summary()

    model1 = Linknet(backbone_name='resnet18', input_shape=(512, 512, 3), classes=2)
    model1.summary()
    plot_model(model1, to_file='./linknet.png', show_shapes=True)
