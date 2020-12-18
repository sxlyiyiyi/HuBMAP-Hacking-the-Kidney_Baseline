import tensorflow as tf
from absl import logging, flags, app
from absl.flags import FLAGS
from utils.data_aug import get_dataset
from utils.util import learning_rate_config, config_optimizer, add_regularization
from utils.score import SegmentationMetric
from utils.visualization import vis_segmentation
from utils.loss import get_loss
from models.model_summary import get_model
import timeit
import numpy as np
from tqdm import tqdm
import cv2
import os

# region 参数配置
# region # 数据集配置
flags.DEFINE_string('train_data', './dataset/record/train.record*', '训练数据集路径')
flags.DEFINE_string('val_data', './dataset/record/val.record*', '验证数据集路径')
flags.DEFINE_integer('batch_size', 6, '每批次大小')
flags.DEFINE_list('size', [1440, 1440], '图像尺寸h, w, 32的整数倍')
flags.DEFINE_integer('num_classes', 2, '类别数')
flags.DEFINE_integer('epochs', 3000, 'epoch')
# endregion

# region # 模型配置
flags.DEFINE_string('model_name', 'pspnet', '模型名称')
flags.DEFINE_string('backbone_name', 'seresnext50', 'backbone名称')
flags.DEFINE_integer('channel', 3, '输入图片通道数')
flags.DEFINE_string('activation', 'sigmoid', '输出层激活函数类型')
flags.DEFINE_string('encoder_weights', 'imagenet', 'backbone预训练参数')
flags.DEFINE_bool('freeze_encoder', False, '冻结backbone参数')

# endregion

# region # 保存与可视化配置
flags.DEFINE_list('label_name', ['yang', 'yin'], '类别名称')
flags.DEFINE_bool('transfor_learning', False, '是否进行迁移学习')
flags.DEFINE_string('tf_dir', './checkpoint/pretraining/BiseNetv2.tf', '预训练模型路径')
flags.DEFINE_integer('weights_num_classes', 19, '预训练模型类别数')

# endregion

# region # 损失函数配置
flags.DEFINE_string('loss_type', 'ce_loss', 'loss类型')
flags.DEFINE_float('weight_decay', 5e-5, '正则化项系数')
flags.DEFINE_integer('top_k', 6554, 'OHEM的top_k')
flags.DEFINE_bool('ignore', False, '是否包含255的忽略像素')
flags.DEFINE_float('alpha', 0.25, 'focal loss的alpha')
flags.DEFINE_float('gamma', 0.8, 'focal loss的gamma')
flags.DEFINE_float('label_smoothing', 0.05, '标签平滑')
flags.DEFINE_list('weights', [1, 1, 1, 1, 1, 1, 1, 1], '交叉熵权重')
# endregion

# region # 优化器配置
flags.DEFINE_string('optimizer_name', 'sgd', '优化器名称：Adam, sgd, RMSProp, Momentum')
flags.DEFINE_float('lr_init', 1e-1, '初始学习率')
flags.DEFINE_float('rho', 0.9, 'RMSProp的RHO设置')
flags.DEFINE_float('momentum', 0.9, '动量')
# endregion

# region # 学习率配置
flags.DEFINE_string('lr_type', 'exponential', '学习率配置类型')
flags.DEFINE_integer('lr_decay_steps', 77*5, '学习率衰减步数')
flags.DEFINE_integer('warmup_steps', 77*10, 'warmup步数')
flags.DEFINE_float('lr_decay_rate', 0.96, 'lr_decay_rate')
flags.DEFINE_float('lr_lower_bound', 5e-5, '学习率下限')
flags.DEFINE_float('t_mul', 1.5, '重启time倍率')
flags.DEFINE_float('m_mul', 0.5, '重启最大值倍率')
flags.DEFINE_float('num_periods', 0.5, 'num_periods')
flags.DEFINE_float('alpha1', 0.0, 'alpha')
flags.DEFINE_float('beta1', 0.0001, 'beta')
flags.DEFINE_float('initial_variance', 1.0, 'initial_variance')
flags.DEFINE_float('variance_decay', 0.55, 'variance_decay')
flags.DEFINE_list('boundaries', [5000, 8000, 10000, 12000, 14000, 16000, 18000, 19000, 20000], 'boundaries')
flags.DEFINE_list('values', [1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5], 'values')
# endregion
# endregion


def train(_argv):

    # region # 显卡配置
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # endregion

    # region # 数据集加载
    train_dataset = get_dataset(FLAGS.train_data, FLAGS.size, FLAGS.batch_size, is_training=True)
    # val_dataset = get_dataset(FLAGS.val_data, FLAGS.size)

    for batch, (images, labels) in enumerate(train_dataset):
        for i in range(FLAGS.batch_size):
            img = np.array(images[i, :, :, :] * 255).astype(np.int64)
            label = np.array(labels[i, :, :, 0]).astype(np.int64)
            x = np.unique(label)
            vis_segmentation(img, label, label_names=FLAGS.label_name)
    # endregion

    # region # 模型搭建和损失函数配置
    model = get_model(FLAGS.model_name, FLAGS.num_classes, FLAGS.size, FLAGS.batch_size, FLAGS.channel,
                      FLAGS.backbone_name, FLAGS.encoder_weights, FLAGS.activation, FLAGS.freeze_encoder)

    model = add_regularization(model, tf.keras.regularizers.l2(FLAGS.weight_decay))
    model.summary()

    loss = get_loss(FLAGS.loss_type, FLAGS.num_classes, FLAGS.weights, FLAGS.top_k, FLAGS.alpha, FLAGS.gamma,
                    FLAGS.label_smoothing, FLAGS.ignore)
    # endregion

    # region # 优化器和学习率配置
    lr = tf.Variable(FLAGS.lr_init)
    learning_rate = learning_rate_config(FLAGS, FLAGS.lr_init, FLAGS.lr_decay_steps)

    # warmup策略
    def lr_with_warmup(global_steps):
        lr_ = tf.cond(tf.less(global_steps, FLAGS.warmup_steps),
                      lambda: FLAGS.lr_init * tf.cast(global_steps / FLAGS.warmup_steps, tf.float32),
                      lambda: tf.maximum(learning_rate(global_steps - FLAGS.warmup_steps), FLAGS.lr_lower_bound))
        return lr_

    optimizer = config_optimizer(FLAGS, learning_rate=lr)
    # endregion

    # region # 模型保存与恢复
    ckpt_dir = './checkpoint/' + FLAGS.model_name + '_' + FLAGS.backbone_name
    log_dir = ckpt_dir + '/logs'
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    # 使用tf.train.CheckpointManager管理Checkpoint
    manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir, max_to_keep=3,
                                         checkpoint_name="ckpt_f1_score")
    latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)  # 会自动找到最近保存的变量文件
    if latest_checkpoint is not None:
        checkpoint.restore(latest_checkpoint)  # 从文件恢复模型参数
        print("restore ckpt successful!!!")
    # endregion

    # region # 训练与验证静态图
    @tf.function
    def train_one_batch(x, y):
        with tf.GradientTape() as tape:
            # 1、计算模型输出和损失
            pred_o = model(x, training=True)
            # pred_o, l2, l3, l4, l5 = model(x, training=True)
            regularization_loss_out = tf.reduce_sum(model.losses)
            # seg_loss_out = loss(y, pred_o) + 0.1 * (loss(y, l2) + loss(y, l3) + loss(y, l4) + loss(y, l5))
            seg_loss_out = loss(y, pred_o)
            total_loss_out = seg_loss_out + regularization_loss_out
        # 计算梯度以及更新梯度, 固定用法
        grads = tape.gradient(total_loss_out, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total_loss_out, seg_loss_out, pred_o

    @tf.function
    def val_one_batch(x, y):
        # pred_o, _, _, _, _ = model(x, training=False)
        pred_o = model(x, training=False)
        return pred_o
    # endregion

    # region # 记录器和评价指标
    summary_writer = tf.summary.create_file_writer(log_dir)
    # tf.summary.trace_on(profiler=True)  # 开启Trace（可选）
    # 评价指标
    val_metric = SegmentationMetric(FLAGS.num_classes)
    train_metric = SegmentationMetric(FLAGS.num_classes)
    val_metric.reset()
    train_metric.reset()
    # endregion

    # region # 迭代优化
    for epoch in range(0, FLAGS.epochs):
        # region # 训练集
        lr.assign(lr_with_warmup(optimizer.iterations))  # 必须使用assign才能改变optimizer的lr的值，否则，是个固定值
        for batch, (images_batch, labels_batch) in tqdm(enumerate(train_dataset)):
            total_loss, seg_loss, train_pred = train_one_batch(images_batch, labels_batch)
            # 计算训练集精度
            if epoch % 3 == 0:
                train_out = np.argmax(train_pred, axis=-1)
                for i in range(labels_batch.shape[0]):
                    train_label = np.array(labels_batch[i, :, :, 0]).astype(np.int64)
                    train_metric.addBatch(train_label, train_out[i, :, :])
            # if epoch > 200:
            with summary_writer.as_default():  # 指定记录器
                tf.summary.scalar("train/total_losses", total_loss, step=optimizer.iterations)  # 将当前损失函数的值写入记录器
                tf.summary.scalar("train/segmentation_loss_loss", seg_loss, step=optimizer.iterations)
                # tf.summary.scalar("train/learning_rate", lr, step=optimizer.iterations)
        # endregion

        # region # 验证集
        if epoch % 3 == 0:
            # ----------------------------------------------验证集验证--------------------------------------------------------
            # for batch, (images_batch, labels_batch) in tqdm(enumerate(val_dataset)):
            #     out = val_one_batch(images_batch, labels_batch)
            #     out = np.squeeze(np.argmax(out, axis=-1))
            #     labels_batch = np.array(labels_batch[0, :, :, 0]).astype(np.int64)
            #     val_metric.addBatch(labels_batch, out)

            with summary_writer.as_default():
                # tf.summary.scalar("val/mPA", val_metric.meanPixelAccuracy(), step=epoch)
                # tf.summary.scalar("val_metric/PA1", val_metric.classPixelAccuracy(1), step=epoch)
                # tf.summary.scalar("val/IoU1", val_metric.IoU(1), step=epoch)
                tf.summary.scalar("train_metric/mPA", train_metric.meanPixelAccuracy(), step=epoch)
                tf.summary.scalar("train_metric/IoU1", train_metric.IoU(1), step=epoch)
                tf.summary.scalar("train_metric/dice", train_metric.dice(), step=epoch)
                TRA_PA = train_metric.meanPixelAccuracy()
                # VAL_PA = val_metric.meanPixelAccuracy()
            train_metric.reset()
            val_metric.reset()
        # endregion

        # region # 模型保存
        # 使用CheckpointManager保存模型参数到文件并自定义编号
        manager.save(checkpoint_number=int(epoch))

        # model.save_weights(.FLAGSckpt_dir + '/BiseNetv2.tf')
        # if TRA_PA >= 0.97 or VAL_PA >= 0.97:
        #     os.mkdir(FLAGS.ckpt_dir + '//epoch_{}_val_{:.5f}_train_{:.5f}'.format(epoch, VAL_PA, TRA_PA))
        #     tf.keras.models.save_model(
        #         model, FLAGS.ckpt_dir + '//epoch_{}_val_{:.5f}_train_{:.5f}'.format(epoch, VAL_PA, TRA_PA))
        # endregion
    # endregion


if __name__ == '__main__':
    app.run(train)
