import tensorflow as tf


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

