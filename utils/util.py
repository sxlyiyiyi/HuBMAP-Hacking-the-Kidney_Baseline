import cv2
import tensorflow as tf
import numpy as np
from absl import logging, flags
from absl.flags import FLAGS
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dense

class_names = ['background', 'Oyster', 'Abalone']


# region # 优化器设置
def config_optimizer(cfg, learning_rate):
    # Adam算法，自适应学习率
    if cfg.TRAIN.OPTIMIZER == "Adam":
        return tf.keras.optimizers.Adam(learning_rate)
    # 梯度下降法
    elif cfg.TRAIN.OPTIMIZER == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=cfg.TRAIN.MOMENTUM)
    # RMSProp算法，自适应学习率
    elif cfg.TRAIN.OPTIMIZER == "RMSProp":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=cfg.TRAIN.RHO, momentum=cfg.TRAIN.MOMENTUM)
    # 动量优化法,一般动量momentum取0.9
    elif cfg.TRAIN.OPTIMIZER == "Momentum":
        return tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise ValueError('Unsupported optimizer type!')
# endregion


def learning_rate_config(cfg):
    learning_rate_init = cfg.SCHEDULER.LR_INIT
    lr_decay_steps = cfg.SCHEDULER.LR_DECAY_STEPS
    # 指数衰减
    if cfg.SCHEDULER.LR_TYPE == "exponential":
        lr_tmp = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate_init,
                                                                decay_steps=lr_decay_steps,
                                                                decay_rate=cfg.SCHEDULER.LR_DECAY_RATE,
                                                                name='exponential_learning_rate')

        return lr_tmp

    # 余弦衰减
    elif cfg.SCHEDULER.LR_TYPE == 'cosine_decay':
        lr_tmp = tf.keras.experimental.CosineDecay(initial_learning_rate=learning_rate_init,
                                                   decay_steps=lr_decay_steps,
                                                   alpha=cfg.SCHEDULER.LR_LOWER_BOUND,
                                                   name='cosine_decay_learning_rate')
        return lr_tmp

    # 重启余弦衰减
    elif cfg.SCHEDULER.LR_TYPE == "cosine_decay_restart":
        lr_tmp = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=learning_rate_init,
                                                           first_decay_steps=lr_decay_steps,
                                                           t_mul=cfg.SCHEDULER.CDR_T_MUL,
                                                           m_mul=cfg.SCHEDULER.CDR_M_MUL,
                                                           name='cosine_decay_learning_rate_restart')
        return lr_tmp

    # 线性余弦衰减
    elif cfg.SCHEDULER.LR_TYPE == "linear_cosine_decay":
        lr_tmp = tf.keras.experimental.LinearCosineDecay(initial_learning_rate=learning_rate_init,
                                                         decay_steps=lr_decay_steps,
                                                         num_periods=cfg.SCHEDULER.NUM_PERIODS,
                                                         alpha=cfg.SCHEDULER.LCD_ALPHA,
                                                         beta=cfg.SCHEDULER.LCD_BETA,
                                                         name='linear_cosine_decay_learning_rate')
        return lr_tmp

    # 噪声线性余弦衰减
    elif cfg.SCHEDULER.LR_TYPE == "noise_linear_cosine_decay":
        lr_tmp = tf.keras.experimental.NoisyLinearCosineDecay(initial_learning_rate=learning_rate_init,
                                                              decay_steps=lr_decay_steps,
                                                              initial_variance=cfg.SCHEDULER.INITIAL_VARIANCE,
                                                              variance_decay=cfg.SCHEDULER.VARIANCE_DECAY,
                                                              num_periods=cfg.SCHEDULER.NUM_PERIODS,
                                                              name='noise_linear_cosine_decay_learning_rate')
        return lr_tmp

    # 倒数衰减
    elif cfg.SCHEDULER.LR_TYPE == 'inverse_time_decay':
        lr_tmp = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=learning_rate_init,
                                                                decay_steps=lr_decay_steps,
                                                                decay_rate=cfg.SCHEDULER.LR_DECAY_RATE,
                                                                name='inverse_time_decay_learning_rate')
        return lr_tmp

    # 固定学习率
    elif cfg.SCHEDULER.LR_TYPE == "fixed":
        return tf.convert_to_tensor(learning_rate_init, name='fixed_learning_rate')

    # 分段常数衰减
    elif cfg.SCHEDULER.LR_TYPE == "piecewise":
        lr_tmp = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=cfg.SCHEDULER.BOUNDARIES,
                                                                      values=cfg.SCHEDULER.SCHEDULER.VALUES,
                                                                      name='piecewise_learning_rate')
        return lr_tmp

    else:
        raise ValueError('Unsupported learning rate type!')


def show_image(img, clas):
    img = (img.numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.putText(img, class_names[int(clas)], (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def show_image(img):
    # img = (img.numpy() * 255).astype(np.uint8)
    img = (img.numpy()).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):
    l2 = regularizer

    def add_l2_regularization(layer_, l2_):
        def _add_l2_regularization():
            if isinstance(layer_, DepthwiseConv2D):
                return l2_(layer_.depthwise_kernel)
            elif isinstance(layer_, Conv2D) or isinstance(layer_, Dense):
                return l2_(layer_.kernel)
        return _add_l2_regularization

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                model.add_loss(add_l2_regularization(layer, l2))

    return model


# 显卡配置
def set_device():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


# checkpoink 配置
def ckpt_manager(cfg, model, logger, opt):
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=opt, net=model)
    manager = tf.train.CheckpointManager(checkpoint, cfg.CKPT_DIR, max_to_keep=3)
    # region # 模型保存与恢复
    # checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    # 使用tf.train.CheckpointManager管理Checkpoint
    # manager = tf.train.CheckpointManager(checkpoint, directory=cfg.CKPT_DIR, max_to_keep=3,
    #                                      checkpoint_name="ckpt")
    # CKPT = tf.train.Checkpoint()
    latest_checkpoint = tf.train.latest_checkpoint(cfg.CKPT_DIR)  # 会自动找到最近保存的变量文件
    if latest_checkpoint is not None:
        checkpoint.restore(latest_checkpoint)  # 从文件恢复模型参数
        logger.info("restore {} successful!!!".format(latest_checkpoint))

    return manager, checkpoint
