- Backbone model:
    - VGGNet
        - VGG16
        - VGG19
    
    - DenseNet:
        - DenseNet121
        - DenseNet169
        - DenseNet201
            
    - ResNet:
        - ResNet18
        - ResNet34
        - ResNet50
        - ResNet101
        - ResNet152
        - resnet50v2
        - resnet101v2
        - resnet152v2
    
    - Inception
        - InceptionV3
        - InceptionResNetV2
    
    - Xception
        - Xception
        
    - MobileNet
        - mobileNet 
        - MobileNet V2
        - MobileNetV3 small
        - MobileNetV3 large
       
    - ResNeXt:
        - ResNeXt50
        - ResNeXt101
    
    - SENet:
        - SEResNet50
        - SEResNet101
        - SEResNet152
        - SEResNeXt50
        - SEResNeXt101
        - SENet154
        
    - NasNet:
        - NasNet large
        - NasNet mobile
    
    - ShuffleNet
        - ShuffleNet
    
    - EfficientNet
        - EfficientNetB0
        - EfficientNetB2
        - EfficientNetB3
        - EfficientNetB4
        - EfficientNetB5
        - EfficientNetB6
        - EfficientNetB7
        - EfficientNetL2
    
The top-k accuracy were obtained using center single crop on the 
2012 ILSVRC ImageNet validation set and may differ from the original ones. 
The input size used was 224x224 (min size 256) for all models except:

 - NASNetLarge 331x331 (352)
 - InceptionV3 299x299 (324)
 - InceptionResNetV2 299x299 (324)
 - Xception 299x299 (324)  

The inference \*Time was evaluated on 500 batches of size 16. 
All models have been tested using same hardware and software. 
Time is listed just for comparison of performance.
          
| Model             |   Acc@1   |   Acc@5   | Time*  | Source                                                       |
| ----------------- | :-------: | :-------: | :----: | ------------------------------------------------------------ |
| vgg16             |   70.79   |   89.74   | 24.95  | [keras](https://github.com/keras-team/keras-applications)    |
| vgg19             |   70.89   |   89.69   | 24.95  | [keras](https://github.com/keras-team/keras-applications)    |
| resnet18          |   68.24   |   88.49   | 16.07  | [mxnet](https://github.com/Microsoft/MMdnn)                  |
| resnet34          |   72.17   |   90.74   | 17.37  | [mxnet](https://github.com/Microsoft/MMdnn)                  |
| resnet50          |   74.81   |   92.38   | 22.62  | [mxnet](https://github.com/Microsoft/MMdnn)                  |
| resnet101         |   76.58   |   93.10   | 33.03  | [mxnet](https://github.com/Microsoft/MMdnn)                  |
| resnet152         |   76.66   |   93.08   | 42.37  | [mxnet](https://github.com/Microsoft/MMdnn)                  |
| resnet50v2        |   69.73   |   89.31   | 19.56  | [keras](https://github.com/keras-team/keras-applications)    |
| resnet101v2       |   71.93   |   90.41   | 28.80  | [keras](https://github.com/keras-team/keras-applications)    |
| resnet152v2       |   72.29   |   90.61   | 41.09  | [keras](https://github.com/keras-team/keras-applications)    |
| resnext50         |   77.36   |   93.48   | 37.57  | [keras](https://github.com/keras-team/keras-applications)    |
| resnext101        |   78.48   |   94.00   | 60.07  | [keras](https://github.com/keras-team/keras-applications)    |
| densenet121       |   74.67   |   92.04   | 27.66  | [keras](https://github.com/keras-team/keras-applications)    |
| densenet169       |   75.85   |   92.93   | 33.71  | [keras](https://github.com/keras-team/keras-applications)    |
| densenet201       |   77.13   |   93.43   | 42.40  | [keras](https://github.com/keras-team/keras-applications)    |
| inceptionv3       |   77.55   |   93.48   | 38.94  | [keras](https://github.com/keras-team/keras-applications)    |
| xception          |   78.87   |   94.20   | 42.18  | [keras](https://github.com/keras-team/keras-applications)    |
| inceptionresnetv2 |   80.03   |   94.89   | 54.77  | [keras](https://github.com/keras-team/keras-applications)    |
| seresnet18        |   69.41   |   88.84   | 20.19  | [pytorch](https://github.com/Cadene/pretrained-models.pytorch) |
| seresnet34        |   72.60   |   90.91   | 22.20  | [pytorch](https://github.com/Cadene/pretrained-models.pytorch) |
| seresnet50        |   76.44   |   93.02   | 23.64  | [pytorch](https://github.com/Cadene/pretrained-models.pytorch) |
| seresnet101       |   77.92   |   94.00   | 32.55  | [pytorch](https://github.com/Cadene/pretrained-models.pytorch) |
| seresnet152       |   78.34   |   94.08   | 47.88  | [pytorch](https://github.com/Cadene/pretrained-models.pytorch) |
| seresnext50       |   78.74   |   94.30   | 38.29  | [pytorch](https://github.com/Cadene/pretrained-models.pytorch) |
| seresnext101      |   79.88   |   94.87   | 62.80  | [pytorch](https://github.com/Cadene/pretrained-models.pytorch) |
| senet154          |   81.06   |   95.24   | 137.36 | [pytorch](https://github.com/Cadene/pretrained-models.pytorch) |
| nasnetlarge       | **82.12** | **95.72** | 116.53 | [keras](https://github.com/keras-team/keras-applications)    |
| nasnetmobile      |   74.04   |   91.54   | 27.73  | [keras](https://github.com/keras-team/keras-applications)    |
| mobilenet         |   70.36   |   89.39   | 15.50  | [keras](https://github.com/keras-team/keras-applications)    |
| mobilenetv2       |   71.63   |   90.35   | 18.31  | [keras](https://github.com/keras-team/keras-applications)    |


### Weights

| Name                     | Classes |   Models   |
| ------------------------ | :-----: | :--------: |
| 'imagenet'               |  1000   | all models |
| 'imagenet11k-place365ch' |  11586  |  resnet50  |
| 'imagenet11k'            |  11221  | resnet152  |
       