import tensorflow as tf
import pathlib
from utils.data_aug import get_dataset
from utils.util import learning_rate_config, config_optimizer, add_regularization
from utils.score import SegmentationMetric
from utils.visualization import vis_segmentation
from utils.loss import get_loss
from models.model_summary import create_model
import numpy as np
from tqdm import tqdm
from config import cfg
from utils.util import set_device, ckpt_manager
from utils.logger import create_logger


def train():
    set_device()
    output_dir = pathlib.Path(cfg.LOG_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = create_logger(name=__name__,
                           output_dir=output_dir,
                           filename='log.txt')
    # 数据集加载
    train_dataset = get_dataset(cfg.DATASET.TRAIN_DATA, cfg, is_training=True)
    val_dataset = get_dataset(cfg.DATASET.VAL_DATA, cfg)

    for batch, (images, labels) in enumerate(train_dataset):
        for i in range(cfg.TRAIN.BATCH_SIZE):
            img = np.array(images[i, :, :, :] * 255).astype(np.int64)
            label = np.array(labels[i, :, :, 0]).astype(np.int64)
            vis_segmentation(img, label, label_names=cfg.DATASET.LABELS)

    # 模型搭建和损失函数配置
    model = create_model(cfg, name=cfg.MODEL_NAME, backbone=cfg.BACKBONE_NAME)
    model = add_regularization(model, tf.keras.regularizers.l2(cfg.LOSS.WEIGHT_DECAY))
    model.summary()

    loss = get_loss(cfg, cfg.LOSS.TYPE)

    # 优化器和学习率配置
    lr = tf.Variable(cfg.SCHEDULER.LR_INIT)
    learning_rate = learning_rate_config(cfg)

    # warmup策略
    def lr_with_warmup(global_steps):
        lr_ = tf.cond(tf.less(global_steps, cfg.SCHEDULER.WARMUP_STEPS),
                      lambda: cfg.SCHEDULER.LR_INIT * tf.cast((global_steps + 1) / cfg.SCHEDULER.WARMUP_STEPS, tf.float32),
                      lambda: tf.maximum(learning_rate(global_steps - cfg.SCHEDULER.WARMUP_STEPS), cfg.SCHEDULER.LR_LOWER_BOUND))
        return lr_

    optimizer = config_optimizer(cfg, learning_rate=lr)

    # 模型保存与恢复
    manager, ckpt = ckpt_manager(cfg, model, logger, optimizer)

    # 训练与验证静态图
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

    # region # 记录器和评价指标
    summary_writer = tf.summary.create_file_writer(cfg.LOG_DIR)
    # tf.summary.trace_on(profiler=True)  # 开启Trace（可选）
    # 评价指标
    val_metric = SegmentationMetric(cfg.DATASET.N_CLASSES)
    train_metric = SegmentationMetric(cfg.DATASET.N_CLASSES)
    val_metric.reset()
    train_metric.reset()
    # endregion

    # region # 迭代优化
    for _ in range(int(ckpt.step), cfg.TRAIN.EPOCHS):
        # region # 训练集
        ckpt.step.assign_add(1)
        lr.assign(lr_with_warmup(optimizer.iterations))  # 必须使用assign才能改变optimizer的lr的值，否则，是个固定值
        for batch, (images_batch, labels_batch) in tqdm(enumerate(train_dataset)):
            total_loss, seg_loss, train_pred = train_one_batch(images_batch, labels_batch)
            # 计算训练集精度
            if int(ckpt.step) % cfg.TRAIN.SNAP_SHOT == 1:
                train_out = np.argmax(train_pred, axis=-1)
                for i in range(labels_batch.shape[0]):
                    train_label = np.array(labels_batch[i, :, :, 0]).astype(np.int64)
                    train_metric.addBatch(train_label, train_out[i, :, :])
            # if epoch > 200:
            with summary_writer.as_default():  # 指定记录器
                tf.summary.scalar("train/total_losses", total_loss, step=optimizer.iterations)  # 将当前损失函数的值写入记录器
                tf.summary.scalar("train/segmentation_loss_loss", seg_loss, step=optimizer.iterations)
                tf.summary.scalar("train/learning_rate", lr, step=optimizer.iterations)
        # endregion

        # region # 验证集
        if int(ckpt.step) % cfg.TRAIN.SNAP_SHOT == 1:
            # ----------------------------------------------验证集验证--------------------------------------------------------
            for batch, (images_batch, labels_batch) in tqdm(enumerate(val_dataset)):
                out = val_one_batch(images_batch, labels_batch)
                out = np.squeeze(np.argmax(out, axis=-1))
                labels_batch = np.array(labels_batch[0, :, :, 0]).astype(np.int64)
                val_metric.addBatch(labels_batch, out)

            with summary_writer.as_default():
                tf.summary.scalar("val_metric/mPA", val_metric.meanPixelAccuracy(), step=int(ckpt.step))
                tf.summary.scalar("val_metric/dice", val_metric.dice(), step=int(ckpt.step))
                tf.summary.scalar("val_metric/IoU1", val_metric.IoU(1), step=int(ckpt.step))
                tf.summary.scalar("val_metric/mIoU", val_metric.mIoU(), step=int(ckpt.step))
                tf.summary.scalar("train_metric/mPA", train_metric.meanPixelAccuracy(), step=int(ckpt.step))
                tf.summary.scalar("train_metric/mIoU", train_metric.mIoU(), step=int(ckpt.step))
                tf.summary.scalar("train_metric/dice", train_metric.dice(), step=int(ckpt.step))
                tf.summary.scalar("train_metric/IoU1", train_metric.IoU(1), step=int(ckpt.step))
                # VAL_PA = val_metric.meanPixelAccuracy()
                logger.info('__EPOCH_{}__: TRAIN_mIoU: {:.5f}, TRAIN_mPA: {:.5f}, TRAIN_dice: {:.5f}; '
                            'VAL_mIoU: {:.5f}, VAL_mPA: {:.5f}, VAL_dice: {:.5f}'
                            .format(int(ckpt.step), train_metric.mIoU(), train_metric.meanPixelAccuracy(), train_metric.dice(),
                                    val_metric.mIoU(), val_metric.meanPixelAccuracy(), val_metric.dice()))
            train_metric.reset()
            val_metric.reset()
        # endregion

        # region # 模型保存
        # 使用CheckpointManager保存模型参数到文件并自定义编号
        manager.save(checkpoint_number=int(ckpt.step))

        # model.save_weights(.FLAGSckpt_dir + '/BiseNetv2.tf')
        # if TRA_PA >= 0.97 or VAL_PA >= 0.97:
        #     os.mkdir(FLAGS.ckpt_dir + '//epoch_{}_val_{:.5f}_train_{:.5f}'.format(epoch, VAL_PA, TRA_PA))
        #     tf.keras.models.save_model(
        #         model, FLAGS.ckpt_dir + '//epoch_{}_val_{:.5f}_train_{:.5f}'.format(epoch, VAL_PA, TRA_PA))
        # endregion
    # endregion


if __name__ == '__main__':
    train()