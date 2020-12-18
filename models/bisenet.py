from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, multiply
from tensorflow.keras.layers import Concatenate, Add, Input
from tensorflow.keras.models import Model
import tensorflow as tf

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


def ConvBnRelu(x, filters=64, kernel_size=3, strides=1, padding='same', activation='relu'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x


def ARM(inputs, filters):
    feat = ConvBnRelu(inputs, filters, 3, 1)
    atten = tf.reduce_mean(feat, [1, 2], keepdims=True)
    atten = ConvBnRelu(atten, filters=filters, kernel_size=1, activation='sigmoid')
    out = multiply([feat, atten])
    return out


def FFM(input_f, input_s, filters):
    x = Concatenate(axis=-1)([input_f, input_s])
    x = ConvBnRelu(x, filters=filters, kernel_size=3, strides=1)
    branch_1 = tf.reduce_mean(x, [1, 2], keepdims=True)
    branch_1 = Conv2D(filters, 1, padding='same', use_bias=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Conv2D(filters, 1, padding='same', use_bias=False)(branch_1)
    branch_1 = Activation('sigmoid')(branch_1)

    branch_1 = multiply([x, branch_1])
    x = Add()([x, branch_1])

    return x


def SptialPath(inputs, ):
    x = ConvBnRelu(inputs, 64, 7, strides=2)
    x = ConvBnRelu(x, 64, 3, strides=2)
    x = ConvBnRelu(x, 64, 3, strides=2)
    x = ConvBnRelu(x, 128, 1, strides=1)

    return x


def ContextPath(input_shape,
                backbone_name,
                encoder_weights='imagenet'):
    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            weights=encoder_weights,
                            include_top=False)

    inputs = backbone.input

    skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name]

    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l for l in skip_connections])
    feat8 = backbone.layers[skip_connection_idx[1]].output
    feat16 = backbone.layers[skip_connection_idx[0]].output
    feat32 = backbone.output

    global_channels = tf.reduce_mean(feat32, [1, 2], keepdims=True)
    global_channels = ConvBnRelu(global_channels, 128, 1, 1)

    feat32 = ARM(feat32, 128)
    feat16 = ARM(feat16, 128)
    feat32 = feat32 + global_channels
    feat32_up = tf.image.resize(feat32, size=[tf.shape(feat16)[1], tf.shape(feat16)[2]], method='bilinear')
    feat32_up = ConvBnRelu(feat32_up, 128, 3, 1)

    feat16 = feat32_up + feat16
    feat16_up = tf.image.resize(feat16, size=[tf.shape(feat8)[1], tf.shape(feat8)[2]], method='bilinear')
    feat16_up = ConvBnRelu(feat16_up, 128, 3, 1)

    return inputs, feat16_up, feat32_up


def BiseNetv1(n_classes,
              input_shape,
              backbone_name='resnet18',
              encoder_weights='imagenet',
              activation='sigmoid',
              output_aux=False):

    inputs, feat_cp8, feat_cp16 = ContextPath(input_shape, backbone_name, encoder_weights=encoder_weights)
    feat_sp = SptialPath(inputs)
    feat_fuse = FFM(feat_sp, feat_cp8, 256)

    feat_out = ConvBnRelu(feat_fuse, 256, 3, 1)
    feat_out = Conv2D(n_classes, 1, 1, padding='same')(feat_out)
    feat_out = tf.image.resize(feat_out, size=[tf.shape(inputs)[1], tf.shape(inputs)[2]], method='bilinear')
    if activation in {'softmax', 'sigmoid'}:
        feat_out = Activation(activation, name=activation)(feat_out)

    if output_aux:
        feat_out16 = ConvBnRelu(feat_cp8, 256, 3, 1)
        feat_out16 = Conv2D(n_classes, 1, 1, padding='same')(feat_out16)
        feat_out16 = tf.image.resize(feat_out16, size=[tf.shape(inputs)[1], tf.shape(inputs)[2]], method='bilinear')

        feat_out32 = ConvBnRelu(feat_cp16, 256, 3, 1)
        feat_out32 = Conv2D(n_classes, 1, 1, padding='same')(feat_out32)
        feat_out32 = tf.image.resize(feat_out32, size=[tf.shape(inputs)[1], tf.shape(inputs)[2]], method='bilinear')

        if activation in {'softmax', 'sigmoid'}:
            feat_out16 = Activation(activation)(feat_out16)
            feat_out32 = Activation(activation)(feat_out32)
        return Model(inputs, (feat_out, feat_out16, feat_out32))

    return Model(inputs, feat_out)


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


def get_backbone(name, *args, **kwargs):
    return backbones[name](*args, **kwargs)


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


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    import os

    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

    model1 = BiseNetv1(2, (512, 512), 'resnet18', output_aux=False)
    model1.summary()
    plot_model(model1, to_file='./bisenetv2.png', show_shapes=True)