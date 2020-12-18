"""
    用于滑动窗口裁剪图像
"""
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import os
import tifffile as tiff
from argparse import ArgumentParser
from multiprocessing import Pool
from absl import logging, flags, app
from absl.flags import FLAGS

Image.MAX_IMAGE_PIXELS = 1000000000000

flags.DEFINE_string('image_dir', '../dataset/train', '验证数据集路径')
flags.DEFINE_string('mask_dir', '../dataset/masks', '裁剪后数据保存路径')
flags.DEFINE_string('vis_dir', '../dataset/visual', '裁剪后数据保存路径')
flags.DEFINE_string('csv_dir', '../dataset/hubmap-kidney-segmentation/train.csv', 'train_csv')
flags.DEFINE_string('tiff_dir', './hubmap-kidney-segmentation/train/', 'tiff')


# RLE编码转图像
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start:start + length] = 1 + m
    return img.reshape(shape).T


# 图像转rle编码
def mask2rle(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs


def crop(_cnt, _crop_image, _mask_label, _img):
    image_name = os.path.join(FLAGS.image_dir, _img + "_" + str(_cnt) + ".png")
    mask_name = os.path.join(FLAGS.mask_dir, _img + "_" + str(_cnt) + ".png")
    vis_name = os.path.join(FLAGS.vis_dir, _img + "_" + str(_cnt) + ".png")
    _crop_image = cv2.cvtColor(_crop_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_name, _crop_image)
    cv2.imwrite(mask_name, _mask_label)
    cv2.imwrite(vis_name, _mask_label * 255)

    return image_name


def creat_dir():
    if not os.path.isdir(FLAGS.image_dir):
        os.makedirs(FLAGS.image_dir)
    if not os.path.isdir(FLAGS.mask_dir):
        os.makedirs(FLAGS.mask_dir)
    if not os.path.isdir(FLAGS.vis_dir):
        os.makedirs(FLAGS.vis_dir)


def data_cut(_argv):
    stride = 900
    target_size = (1024, 1024)
    s_th = 40  # saturation blancking threshold
    p_th = 200 * target_size[0] // 256  # threshold for the minimum number of pixels
    creat_dir()
    df_masks = pd.read_csv(FLAGS.csv_dir).set_index('id')

    x_tot, x2_tot = [], []
    for index, encs in tqdm(df_masks.iterrows(), total=len(df_masks)):
        image = tiff.imread(os.path.join(FLAGS.tiff_dir, index + '.tiff'))
        if len(image.shape) == 5:
            image = np.transpose(image.squeeze(), (1, 2, 0))
        mask = enc2mask(encs, (image.shape[1], image.shape[0]))

        cnt = 0
        # 填充外边界至裁剪尺寸整数倍
        target_w, target_h = target_size
        h, w = image.shape[0], image.shape[1]
        new_w = (w // target_w) * target_w if (w % target_w == 0) else (w // target_w + 1) * target_w
        new_h = (h // target_h) * target_h if (h % target_h == 0) else (h // target_h + 1) * target_h
        image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=0)

        # 填充1/2 stride长度的外边框
        image = cv2.copyMakeBorder(image, stride // 2, stride // 2, stride // 2, stride // 2, cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, stride // 2, stride // 2, stride // 2, stride // 2, cv2.BORDER_CONSTANT,
                                  value=0)

        h, w = image.shape[0], image.shape[1]

        for i in tqdm(range(h // stride - 1)):
            for j in range(w // stride - 1):
                topleft_x = j * stride
                topleft_y = i * stride
                crop_image = image[topleft_y:topleft_y + target_h, topleft_x:topleft_x + target_w]
                crop_label = mask[topleft_y:topleft_y + target_h, topleft_x:topleft_x + target_w]

                if crop_image.shape[:2] != (target_h, target_h):
                    print(topleft_x, topleft_y, crop_image.shape)
                #
                hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
                _, s, _ = cv2.split(hsv)
                if (s > s_th).sum() <= p_th or crop_image.sum() <= p_th:
                    pass
                else:
                    crop(cnt, crop_image, crop_label, index)
                    cnt += 1
                    x_tot.append((crop_image / 255.0).reshape(-1, 3).mean(0))
                    x2_tot.append(((crop_image / 255.0) ** 2).reshape(-1, 3).mean(0))

    # image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
    print('mean:', img_avr, ', std:', img_std)


if __name__ == "__main__":
    app.run(data_cut)