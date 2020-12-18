- 支持的损失函数
    - CE_Loss: 用于多分类（包括二分类）的交叉熵损失函数
    - W_CELoss: 带权重的交叉熵损失，可以平衡原本数量
    - DiceLoss: 用于多分类（包括二分类）的Dice Loss
    - CE_Dice_Loss: 用于多分类的Dice Loss和多元交叉熵Loss加权
    - FocalLoss: 用于多分类（包括二分类）的Focal Loss
    - B_FocalLoss: 用于二分类的Focal Loss
    - Tversky_Loss: 用于多分类（包括二分类）的Tversky Loss
    - Focal_Tversky_Loss: 用于多分类（包含二分类）的Focal Loss和Tversky_loss加权
    - Generalized_Dice_Loss: 改善Dice Loss，将多个类别的Dice Loss进行整合，使用一个参数作为分割结果的量化指标
    - OHEM_CE_Loss: OHEN交叉熵损失函数
    - Jaccard_Loss: Jaccard/IoU 损失函数
    - CE_Jaccard_Loss: Jaccard/IoU 损失函数和多元交叉熵Loss加权

- 支持的评价指标
    - cPA
    - mPA
    - mIoU
    - IoU
    - FWIoU
    - dice
    
- 支持的学习率配置
    - exponential: 指数衰减
    - cosine_decay: 余弦衰减
    - cosine_decay_restart: 重启余弦衰减
    - linear_cosine_decay: 线性余弦衰减
    - noise_linear_cosine_decay: 噪声线性余弦衰减
    - inverse_time_decay: 倒数衰减
    - fixed: 固定学习率
    - piecewise: 分段常数衰减
    
- 支持的优化器:
    - Adam: Adam算法，自适应学习率
    - sgd: 梯度下降法
    - RMSProp: RMSProp算法，自适应学习率
    - Momentum: 动量优化法,一般动量momentum取0.9
    
    
- util.py
    - add_regularization: 添加正则损失
    