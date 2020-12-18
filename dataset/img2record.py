import os
import numpy as np
from PIL import Image
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from tqdm import trange
import contextlib2

flags.DEFINE_string('dataset', '../dataset/train', '训练集路径')
flags.DEFINE_string('train_record_path', '../dataset/record/train.record', '训练集存储路径')
flags.DEFINE_string('val_record_path', '../dataset/record/val.record', '验证集存储路径')
flags.DEFINE_string('test_record_path', '../dataset/record/test.record', '验证集存储路径')
flags.DEFINE_string('train_txt', '../dataset/train.txt', '训练集txt路径')
flags.DEFINE_string('val_txt', '../dataset/val.txt', '验证集txt路径')


def get_files(file_dir, ratio=0.9):
    images = []
    labels = []

    for image in os.listdir(file_dir):
        image_path = os.path.join(file_dir, image)
        label_path = file_dir[0:-5] + 'masks/' + image[0:-4] + '.png'
        images.append(image_path)
        labels.append(label_path)

    # 按比例划分训练集和验证集
    s1 = np.int(len(images) * ratio)
    # 组合训练集和验证集
    tmp_list = np.array([images, labels])
    tmp_list = tmp_list.transpose()
    # 随机排列
    np.random.shuffle(tmp_list)
    tra_image_list = list(tmp_list[:s1, 0])
    tra_label_list = list(tmp_list[:s1, 1])
    val_image_list = list(tmp_list[s1:, 0])
    val_label_list = list(tmp_list[s1:, 1])

    print("There are %d tra_image_list \nThere are %d tra_label_list \n"
          "There are %d val_image_list \nThere are %d val_label_list \n"
          % (len(tra_image_list), len(tra_label_list), len(val_image_list), len(val_label_list)))

    train_data = open(FLAGS.train_txt, 'w')
    for i in range(len(tra_label_list)):
        train_data.write(tra_image_list[i] + ' ' + (tra_label_list[i]) + '\n')
    train_data.close()

    val_data = open(FLAGS.val_txt, 'w')
    for i in range(len(val_label_list)):
        val_data.write(val_image_list[i] + ' ' + str(val_label_list[i]) + '\n')
    val_data.close()
    #
    # 注意，image_list里面其实存的图片文件的路径
    return tra_image_list, tra_label_list, val_image_list, val_label_list


def image2tfrecord(image_list, filename):
    # 生成字符串型的属性
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # 生成整数型的属性
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    print("len=", len(image_list))
    # 创建一个writer来写TFRecord文件，filename是输出TFRecord文件的地址

    writer = tf.io.TFRecordWriter(filename)
    len2 = len(image_list)
    for i in trange(len2):
        # 读取图片并解码
        image = Image.open(image_list[i])
        # image = image.resize((256, 256))
        # 转化为原始字节
        image_bytes = image.tobytes()

        # 创建字典
        features = {}
        # 用bytes来存储image
        features['image_raw'] = _bytes_feature(image_bytes)
        # 用bytes来存储label
        features['file_name'] = _int64_feature(int(image_list[i].split('\\')[-1][0:-4]))
        # 将所有的feature合成features
        tf_features = tf.train.Features(feature=features)
        # 将样本转成Example Protocol Buffer，并将所有的信息写入这个数据结构
        tf_example = tf.train.Example(features=tf_features)
        # 序列化样本
        tf_serialized = tf_example.SerializeToString()
        # 将序列化的样本写入trfrecord
        writer.write(tf_serialized)
    writer.close()


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.
  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards
  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def create2records(image_list, label_list, output_path, num_shards):
    # 生成字符串型的属性
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # 生成整数型的属性
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        for idx, image in enumerate(image_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(image_list))
            image = Image.open(image)
            image = image.resize((512, 512))
            label = Image.open(label_list[idx])
            label = label.resize((512, 512), 0)
            # 转化为原始字节
            image_bytes = image.tobytes()
            label_bytes = label.tobytes()

            # 创建字典
            features = {}
            # 用bytes来存储image
            features['image_raw'] = _bytes_feature(image_bytes)
            # 用bytes来存储label
            features['label_image'] = _bytes_feature(label_bytes)
            # 将所有的feature合成features
            tf_features = tf.train.Features(feature=features)
            # 将样本转成Example Protocol Buffer，并将所有的信息写入这个数据结构
            tf_example = tf.train.Example(features=tf_features)
            # 序列化样本
            tf_serialized = tf_example.SerializeToString()

            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_serialized)
        logging.info('Finished writing %d images', idx + 1)


def main(_grgv):
    train_images, train_labels, val_images, val_labels = get_files(FLAGS.dataset)
    # image2tfrecord(test_list, FLAGS.test_record_path)
    create2records(train_images, train_labels, FLAGS.train_record_path, 20)
    create2records(val_images, val_labels, FLAGS.val_record_path, 1)


if __name__ == '__main__':

    app.run(main)