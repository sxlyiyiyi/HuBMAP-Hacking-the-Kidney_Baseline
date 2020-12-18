

mean: [0.65459856 0.48386562 0.69428385] , std: [0.15167958 0.23584107 0.13146145]
mean: [0.66174381 0.49755032 0.69969819] , std: [0.16262825 0.24692947 0.14005639]


- 数据预处理
    - plan1: 以 256 的步长滑动裁剪尺寸为 256 * 256 的图片 , 
    - plan2: 以 256 的步长滑动裁剪尺寸为 512 * 512 的图片
    - plan3: 以 768 的步长滑动裁剪尺寸为 768 * 768 的图片
    - plan4: 以 900 的步长滑动裁剪尺寸为 1024 * 1024 的图片

- 数据增广
    - 随机缩放裁剪 0.75 - 1.5
    - 随机旋转  20
    - 随机上下左右翻转
    - 亮度 0.2   prob: 0.7
    - 对比度 0.2  prob:0.7
    - 饱和度 0.2

- 损失函数：
    - 交叉熵损失
    - focal loss
    - b_dice loss
    - w_b_dice loss
    - 标签平滑
    - bce + focal

- 模型：
    - seresnext_unet
    - efficientnet_b3_unet
    - hrnetw48 
    
- TTA：
    - 上下左右翻转
    - 多尺度
    - 多模型投票
    
- 伪标签？


- 先分类再分割？