from models.unet2s import Xnet_s
from models.unet2plus import Xnet
from models.deeplabv3plus import Deeplabv3
from models.pspnet import PSPNet
from models.u2net import U2net
from models.hrnet import seg_hrnet
from models.unet3plus import Unet3plus
from models.bisenetv2 import BiseNetV2
from models.lednet import LEDNet
from models.bisenet import BiseNetv1
from models.fpn import FPN
from models.linknet import Linknet
from models.nestnet import Nestnet
from models.unet import Unet


def create_model(cfg, name, backbone):
    """
        freeze_encoder:
        encoder_weights:
        name: 'bisenetv1', 'bisenetv2', 'deeplabv3plus', 'fpn', 'hrnet_w18', 'hrnet_w32', 'hrnet_w48', 'lednet',
              'linknet', 'nestnet', 'pspnet', 'u2net', 'unet', 'unet2plus', 'unet2s', 'unet3plus'
        classes:
        size:
        batch_size:
        channel:
        backbone: 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50',
                  'resnext101', 'inceptionv3', 'inceptionresnet', 'densenet121', 'densenet169', 'densenet201',
                  'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4',
                  'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'efficientnetl2', 'seresnext50', 'seresnext101',
                  'seresnet50', 'seresnet101', 'seresnet152', 'senet154'
        activation:

    """
    classes = cfg.DATASET.N_CLASSES
    size = cfg.DATASET.SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    channel = cfg.DATASET.CHANNELS
    encoder_weights = cfg.TRAIN.ENCODER_WEIGHTS
    activation = cfg.TRAIN.ACTIVATION
    freeze_encoder = cfg.TRAIN.FREEZE_ENCODER

    if name == 'bisenetv1':
        model = BiseNetv1(cfg, input_shape=(size[0], size[1], channel), backbone_name=backbone,
                          encoder_weights=encoder_weights, activation=activation)
    elif name == 'bisenetv2':
        model = BiseNetV2(classes, size, channel=channel, activation=activation)

    elif name == 'deeplabv3plus':
        model = Deeplabv3(input_shape=(size[0], size[1], channel), classes=classes,
                          backbone=backbone, activation=activation)
    elif name == 'fpn':
        model = FPN(backbone, input_shape=(size[0], size[1], channel), classes=classes, activation=activation,
                    encoder_weights=encoder_weights, encoder_freeze=freeze_encoder)
    elif name.startswith('hrnet'):
        model = seg_hrnet(batch_size, size[0], size[1], channel, classes,
                          activation=activation, hrnet_type=name)
    elif name == 'lednet':
        model = LEDNet(classes, size[0], size[1], channel, activation=activation)
    elif name == 'linknet':
        model = Linknet(backbone, input_shape=(size[0], size[1], channel), encoder_weights=encoder_weights,
                        freeze_encoder=freeze_encoder, classes=classes, activation=activation)
    elif name == 'nestnet':
        model = Nestnet(backbone, input_shape=(size[0], size[1], channel), encoder_weights=encoder_weights,
                        freeze_encoder=freeze_encoder, classes=classes, activation=activation)
    elif name == 'pspnet':
        model = PSPNet(backbone, (size[0], size[1], channel), classes, activation=activation,
                       encoder_weights=encoder_weights, freeze_encoder=freeze_encoder)
    elif name == 'u2net':
        model = U2net((size[0], size[1], channel), classes, activation=activation)
    elif name == 'unet':
        model = Unet(backbone, input_shape=(size[0], size[1], channel), encoder_weights=encoder_weights,
                     freeze_encoder=freeze_encoder, classes=classes, activation=activation)
    elif name == 'unet2plus':
        model = Xnet(backbone, (size[0], size[1], channel), classes=classes, encoder_weights=encoder_weights,
                     freeze_encoder=freeze_encoder, activation=activation)
    elif name == 'unet2s':
        model = Xnet_s(backbone, (size[0], size[1], channel), classes=classes, encoder_weights=encoder_weights,
                       activation=activation, freeze_encoder=freeze_encoder)
    elif name == 'unet3plus':
        model = Unet3plus(name, (size[0], size[1], channel), classes=classes, activation=activation,
                          encoder_weights=encoder_weights, freeze_encoder=freeze_encoder)

    else:
        raise TypeError('Unsupported model type')

    return model
