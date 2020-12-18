import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, add, Activation, average
from tensorflow.keras.layers import MaxPool2D, concatenate, UpSampling2D, Input, Dropout
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


def Unet3plus(backbone_name='vgg16',
              input_shape=(None, None, 3),
              input_tensor=None,
              encoder_weights='imagenet',
              activation='softmax',
              freeze_encoder=False,
              skip_connections='default',
              decoder_filters=[32, 64, 128, 256, 512],
              classes=1):
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
                       decoder_filters,
                       activation)

    # lock encoder weights for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    model._name = 'u-{}'.format(backbone_name)

    return model


def conv_block(inputs, filters, kernel_size=3, strides=1, padding='same'):
    Z = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
    Z = BatchNormalization()(Z)
    A = ReLU()(Z)
    return A


def build_unet(backbone, num_classes, skip_connection_layers, filters, activation='softmax'):
    CatChannels = filters[0]
    CatBlocks = 5
    UpChannels = CatChannels * CatBlocks

    inputs = backbone.input
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])

    h1 = backbone.layers[skip_connection_idx[3]].output
    h2 = backbone.layers[skip_connection_idx[2]].output
    h3 = backbone.layers[skip_connection_idx[1]].output
    h4 = backbone.layers[skip_connection_idx[0]].output
    hd5 = backbone.output

    h1_PT_hd4 = MaxPool2D(strides=(8, 8), padding='same')(h1)
    h1_PT_hd4 = conv_block(h1_PT_hd4, filters[0])
    h2_PT_hd4 = MaxPool2D(strides=(4, 4), padding='same')(h2)
    h2_PT_hd4 = conv_block(h2_PT_hd4, filters[1])
    h3_PT_hd4 = MaxPool2D(strides=(2, 2), padding='same')(h3)
    h3_PT_hd4 = conv_block(h3_PT_hd4, filters[2])
    h4_Cat_hd4 = conv_block(h4, filters[3])
    hd5_UT_hd4 = tf.image.resize(hd5, (tf.shape(h4)[1], tf.shape(h4)[2]), 'bilinear')
    hd5_UT_hd4 = conv_block(hd5_UT_hd4, filters[4])

    # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
    hd4 = concatenate([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4])
    hd4 = conv_block(hd4, UpChannels)

    # stage 3d
    h1_PT_hd3 = MaxPool2D(strides=(4, 4))(h1)
    h1_PT_hd3 = conv_block(h1_PT_hd3, filters[0])
    h2_PT_hd3 = MaxPool2D(strides=(2, 2))(h2)
    h2_PT_hd3 = conv_block(h2_PT_hd3, filters[1])
    h3_Cat_hd3 = conv_block(h3, filters[2])
    hd4_UT_hd3 = tf.image.resize(hd4, (tf.shape(h3)[1], tf.shape(h3)[2]), 'bilinear')
    hd4_UT_hd3 = conv_block(hd4_UT_hd3, UpChannels)
    hd5_UT_hd3 = tf.image.resize(hd5, (tf.shape(h3)[1], tf.shape(h3)[2]), 'bilinear')
    hd5_UT_hd3 = conv_block(hd5_UT_hd3, UpChannels)

    hd3 = concatenate([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3])
    hd3 = conv_block(hd3, UpChannels)

    # stage 2d
    h1_PT_hd2 = MaxPool2D(strides=(2, 2))(h1)
    h1_PT_hd2 = conv_block(h1_PT_hd2, filters[0])
    h2_Cat_hd2 = conv_block(h2, filters[1])
    hd3_UT_hd2 = tf.image.resize(hd3, (tf.shape(h2)[1], tf.shape(h2)[2]), 'bilinear')
    hd3_UT_hd2 = conv_block(hd3_UT_hd2, UpChannels)
    hd4_UT_hd2 = tf.image.resize(hd4, (tf.shape(h2)[1], tf.shape(h2)[2]), 'bilinear')
    hd4_UT_hd2 = conv_block(hd4_UT_hd2, UpChannels)
    hd5_UT_hd2 = tf.image.resize(hd5, (tf.shape(h2)[1], tf.shape(h2)[2]), 'bilinear')
    hd5_UT_hd2 = conv_block(hd5_UT_hd2, UpChannels)

    hd2 = concatenate([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2])
    hd2 = conv_block(hd2, UpChannels)

    # stage 1d
    h1_Cat_hd1 = conv_block(h1, filters[0])
    hd2_UT_hd1 = tf.image.resize(hd2, (tf.shape(h1)[1], tf.shape(h1)[2]), 'bilinear')
    hd2_UT_hd1 = conv_block(hd2_UT_hd1, UpChannels)
    hd3_UT_hd1 = tf.image.resize(hd3, (tf.shape(h1)[1], tf.shape(h1)[2]), 'bilinear')
    hd3_UT_hd1 = conv_block(hd3_UT_hd1, UpChannels)
    hd4_UT_hd1 = tf.image.resize(hd4, (tf.shape(h1)[1], tf.shape(h1)[2]), 'bilinear')
    hd4_UT_hd1 = conv_block(hd4_UT_hd1, UpChannels)
    hd5_UT_hd1 = tf.image.resize(hd5, (tf.shape(h1)[1], tf.shape(h1)[2]), 'bilinear')
    hd5_UT_hd1 = conv_block(hd5_UT_hd1, UpChannels)

    hd1 = concatenate([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1])
    hd1 = conv_block(hd1, UpChannels)  # shape=(None, 320, 320, 320)

    d5 = Conv2D(num_classes, kernel_size=3, activation=None, padding='same', use_bias=False)(hd5)
    d5 = tf.image.resize(d5, (tf.shape(inputs)[1], tf.shape(inputs)[2]), 'bilinear')

    d4 = Conv2D(num_classes, kernel_size=3, activation=None, padding='same', use_bias=False)(hd4)
    d4 = tf.image.resize(d4, (tf.shape(inputs)[1], tf.shape(inputs)[2]), 'bilinear')

    d3 = Conv2D(num_classes, kernel_size=3, activation=None, padding='same', use_bias=False)(hd3)
    d3 = tf.image.resize(d3, (tf.shape(inputs)[1], tf.shape(inputs)[2]), 'bilinear')

    d2 = Conv2D(num_classes, kernel_size=3, activation=None, padding='same', use_bias=False)(hd2)
    d2 = tf.image.resize(d2, (tf.shape(inputs)[1], tf.shape(inputs)[2]), 'bilinear')

    d1 = Conv2D(num_classes, kernel_size=3, activation=None, padding='same', use_bias=False)(hd1)
    d = tf.image.resize(d1, (tf.shape(inputs)[1], tf.shape(inputs)[2]), 'bilinear')
    if activation in {'softmax', 'sigmoid'}:
        d = Activation(activation, name='d1')(d)
        d2 = Activation(activation, name='d2')(d2)
        d3 = Activation(activation, name='d3')(d3)
        d4 = Activation(activation, name='d4')(d4)
        d5 = Activation(activation, name='d5')(d5)
    # d = average([d1, d2, d3, d4, d5])
    # d = Conv2D(num_classes, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
    # d = Activation('sigmoid', name='d')(d)
    model = Model(inputs=inputs, outputs=[d, d2, d3, d4, d5])
    return model


def freeze_model(model):
    for layer in model.layers:
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


if __name__ == '__main__':
    model1 = Unet3plus(backbone_name='efficientnetb0', input_shape=(512, 1024, 3))
    model1.summary()

    # cls_branch=Dropout(0.5)(hd5)
    # cls_branch =Conv2D(2,kernel_size=1,padding='same')(cls_branch)
    # cls_branch=MaxPool2D()(cls_branch )
    # cls_branch =Activation('sigmoid')(cls_branch ) #shape=(None, 10, 10, 1024)
