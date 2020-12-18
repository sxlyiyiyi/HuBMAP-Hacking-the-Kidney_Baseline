from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, LeakyReLU, Dropout, MaxPooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Add, concatenate
from tensorflow.keras.models import Model
import numpy as np
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


def Xnet_s(backbone_name='vgg16',
           input_shape=(None, None, 3),
           input_tensor=None,
           encoder_weights='imagenet',
           activation='softmax',
           freeze_encoder=False,
           skip_connections='default',
           dropout_rate=0.,
           start_neurons=8,
           classes=1):
    """

    Args:
        activation:
        backbone_name: (str) look at list of available backbones.
        input_shape:  (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            'dof' (pre-training on DoF)
        dropout_rate:
        start_neurons:
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        skip_connections: if 'default' is used take default skip connections,
            else provide a list of layer numbers or names starting from top of model
        classes: (int) a number of classes for output

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
    # n_upsample_blocks = len(skip_connections)

    model = build_xnet(backbone,
                       classes,
                       skip_connections,
                       dropout_rate,
                       start_neurons,
                       activation)

    # lock encoder weights for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    model._name = 'x-{}'.format(backbone_name)

    return model


def build_xnet(backbone,
               num_classes,
               skip_connection_layers,
               dropout_rate,
               start_neurons=8,
               activation='softmax'):

    inputs = backbone.input

    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])

    conv1 = backbone.layers[skip_connection_idx[3]].output
    conv2 = backbone.layers[skip_connection_idx[2]].output
    conv3 = backbone.layers[skip_connection_idx[1]].output
    conv4 = backbone.layers[skip_connection_idx[0]].output

    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same", name='conv_middle')(pool4)
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    deconv4_up1 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4)
    deconv4_up2 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up1)
    deconv4_up3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up2)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)

    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    #     uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)  # conv1_2

    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3_up1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3)
    deconv3_up2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3_up1)

    uconv3 = concatenate([deconv3, deconv4_up1, conv3])
    uconv3 = Dropout(dropout_rate)(uconv3)

    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    #     uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    deconv2_up1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv2)

    uconv2 = concatenate([deconv2, deconv3_up1, deconv4_up2, conv2])

    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    #     uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)

    uconv1 = concatenate([deconv1, deconv2_up1, deconv3_up2, deconv4_up3, conv1])

    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    #     uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    #     uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = Dropout(dropout_rate / 2)(uconv0)
    output_layer = Conv2D(num_classes, (1, 1), padding="same")(uconv0)
    if activation in {'softmax', 'sigmoid'}:
        output_layer = Activation(activation, name=activation)(output_layer)

    model = Model(inputs, output_layer)

    return model


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation is True:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])

    return x


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

    model1 = Xnet_s(backbone_name='efficientnetb4', input_shape=(512, 512, 3), classes=2)
    model1.summary()
    plot_model(model1, to_file='./linknet.png', show_shapes=True)
