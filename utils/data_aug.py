import tensorflow as tf
import numpy as np
from utils.imaug.Resize import StepScaleCrop, random_resizedcrop, random_zoom
from utils.imaug.Cutout import random_cutout
from utils.imaug.GridMask import gridmask
from utils.imaug.CutMix import cutmix
from utils.imaug.RnadomErasing import random_erasing


# 基础数据增广,颜色方面（调节对比度，调节亮度，调节Hue，添加饱和度，添加高斯噪声等
@tf.function
def adjust_color(img, mask):
    # 调整对比度
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    # 增加亮度
    img = tf.image.random_brightness(img, max_delta=0.2)
    # 调整色调
    # img = tf.image.random_hue(img, max_delta=0.15)
    # 调整饱和度
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    # 添加高斯噪声
    # noise = tf.random.normal(shape=tf.shape(img), mean=0, stddev=1.0, dtype=tf.float32) / 255
    # img = tf.add(img, noise)
    # 将img限制在 [0, 1]
    img = tf.clip_by_value(img, 0, 1)

    return img, mask


# 随机翻转
@tf.function
def random_flip(img, mask, vertical=False):
    img = tf.image.random_flip_left_right(img)
    if vertical is True:
        img = tf.image.random_flip_up_down(img)

    return img, mask


# tfrecord格式转换
def _parse_tfrecord(tfrecord, cfg):
    x = tf.io.parse_single_example(tfrecord, features={
        # tf.FixedLenFeature解析的结果为一个tensor
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label_image': tf.io.FixedLenFeature([], tf.string),
    })  # 取出包含image和label的feature对象

    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.io.decode_raw(x['image_raw'], tf.uint8)
    # 根据图像尺寸，还原图像
    image = tf.reshape(image, [cfg.DATASET.RAW_SIZE[0], cfg.DATASET.RAW_SIZE[1], cfg.DATASET.CHANNELS])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, cfg.DATASET.SIZE)
    label_image = tf.io.decode_raw(x['label_image'], tf.uint8)
    label = tf.reshape(label_image, [cfg.DATASET.RAW_SIZE[0], cfg.DATASET.RAW_SIZE[1], 1])
    # label = tf.cast(label > 128, tf.float32)
    # cla = tf.cast(x['class'], tf.int32)
    # label = tf.image.convert_image_dtype(label, tf.float32)
    label = tf.image.resize(label, cfg.DATASET.SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, label


# 数据读取
def get_dataset(file, cfg, is_training=False):
    # 获取文件名列表
    files = tf.data.Dataset.list_files(file)
    # 读入原始的TFRecord文件, 获得一个 tf.data.Dataset 数据集对象
    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda x: _parse_tfrecord(x, cfg))

    if is_training:
        dataset = dataset.shuffle(buffer_size=300)
        dataset = dataset.map(adjust_color)
        dataset = dataset.map(lambda x, y: random_flip(x, y, vertical=True))
        dataset = dataset.map(lambda x, y: StepScaleCrop(x, y, size=cfg.DATASET.SIZE, dl=0.7, ul=1.5, step_size=0.1))
        # dataset = dataset.map(lambda x, y: random_zoom(x, y, dl=0.5, ul=1.5))
        # dataset = dataset.map(lambda x, y: gridmask(x, y, d1=100, d2=120, rotate=15, ratio=0.1))
        # dataset = dataset.map(lambda x, y: random_resizedcrop(x, y, size=cfg.DATASET.SIZE, dl=0.5, ul=1.5))
        # dataset = dataset.map(lambda x, y: random_erasing(x, y, probability=0.5, sl=0.02, sh=0.4, r1=0.3))
        dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE)
        # dataset = dataset.map(lambda x, y: cutmix(x, y, cfg.TRAIN.BATCH_SIZE))
        # dataset = dataset.map(lambda x, y: random_cutout(x, y, mask_size=40))
    else:
        dataset = dataset.batch(1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
