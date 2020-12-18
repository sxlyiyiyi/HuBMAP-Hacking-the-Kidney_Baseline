import tensorflow as tf
from models.backbone.efficientnet import (EfficientNetB0,
                                          EfficientNetB1,
                                          EfficientNetB2,
                                          EfficientNetB3,
                                          EfficientNetB4,
                                          EfficientNetB5,
                                          EfficientNetB6,
                                          EfficientNetB7)

from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, Conv2DTranspose
from tensorflow.keras.layers import Dropout, MaxPooling2D, concatenate, Activation
from tensorflow.keras.models import Model


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


def uefficientnet(input_shape, num_classes, dropout_rate, backbone='EfficientNetB4', activation='softmax'):
    if backbone == 'EfficientNetB0':
        Backbone = EfficientNetB0(include_top=False,
                                  weights='imagenet',
                                  input_shape=input_shape)
    elif backbone == 'EfficientNetB1':
        Backbone = EfficientNetB1(include_top=False,
                                  weights='imagenet',
                                  input_shape=input_shape)
    elif backbone == 'EfficientNetB2':
        Backbone = EfficientNetB2(include_top=False,
                                  weights='imagenet',
                                  input_shape=input_shape)
    elif backbone == 'EfficientNetB3':
        Backbone = EfficientNetB3(include_top=False,
                                  weights='imagenet',
                                  input_shape=input_shape)
    elif backbone == 'EfficientNetB4':
        Backbone = EfficientNetB4(include_top=False,
                                  weights='imagenet',
                                  input_shape=input_shape)
    elif backbone == 'EfficientNetB5':
        Backbone = EfficientNetB5(include_top=False,
                                  weights='imagenet',
                                  input_shape=input_shape)
    elif backbone == 'EfficientNetB6':
        Backbone = EfficientNetB6(include_top=False,
                                  weights='imagenet',
                                  input_shape=input_shape)
    else:
        raise ValueError('Unsupportted backbone type!')
    inputs = Backbone.input
    conv1 = Backbone.get_layer('block2a_expand_activation').output
    conv2 = Backbone.get_layer('block3a_expand_activation').output
    conv3 = Backbone.get_layer('block4a_expand_activation').output
    conv4 = Backbone.get_layer('block6a_expand_activation').output
    start_neurons = 8

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
    output_layer = Activation(activation)(output_layer)

    model = Model(inputs, output_layer)

    return model


if __name__ == '__main__':
    # model1 = EfficientNetB4(input_shape=(256, 256, 3))
    model1 = uefficientnet(input_shape=(256, 256, 3), num_classes=19, dropout_rate=0.05, backbone='EfficientNetB4')
    model1.summary()