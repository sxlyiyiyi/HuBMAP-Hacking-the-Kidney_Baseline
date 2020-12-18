import numpy as np
import cv2
import os
import tensorflow as tf
import tifffile as tiff
import pandas as pd
from PIL import Image
from models.model_summary import create_model
from config import cfg
from tqdm import tqdm
from utils.visualization import vis_segmentation
from utils.util import set_device


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


def eval_img():
    tiff_dir = './dataset/hubmap-kidney-segmentation/test'

    test_mask_path = './result/'
    if not os.path.isdir(test_mask_path):
        os.makedirs(test_mask_path)
    target_size = (512, 512)
    stride = 512      # stride 应该大于等于 1/2 target_size, 小于等于target_size
    hstride = stride // 2
    dis_wh = (target_size[0] - stride) // 2
    resize = 0.5

    test = pd.DataFrame(columns=['id', 'predicted'])

    set_device()
    model = create_model(cfg, name=cfg.MODEL_NAME, backbone=cfg.BACKBONE_NAME)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), net=model)
    latest_checkpoint = tf.train.latest_checkpoint(cfg.CKPT_DIR)  # 会自动找到最近保存的变量文件
    if latest_checkpoint is not None:
        print("restore successful")
        checkpoint.restore(latest_checkpoint)  # 从文件恢复模型参数

    @tf.function
    def tta_interfence(y):
        y = y / 255.
        out_ = model(y, training=False)
        # out_, _, _, _, _ = model(y, training=False)
        return out_

    for img in os.listdir(tiff_dir):
        if not img.endswith('tiff'):
            continue
        image_path = os.path.join(tiff_dir, img)
        image = tiff.imread(image_path)
        image = np.asarray(image)
        if len(image.shape) == 5:
            image = np.transpose(image.squeeze(), (1, 2, 0))
        hh, ww = image.shape[0], image.shape[1]
        image = cv2.resize(image, (int(image.shape[1] * resize), int(image.shape[0] * resize)))

        # 填充外边界至步长整数倍
        h, w = image.shape[0], image.shape[1]
        target_w, target_h = target_size
        new_w = (w // target_w) * target_w if (w % target_w == 0) else (w // target_w + 1) * target_w
        new_h = (h // target_h) * target_h if (h % target_h == 0) else (h // target_h + 1) * target_h
        image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, 0)

        # 填充1/2 stride长度的外边框
        new_w, new_h = image.shape[1] + stride, image.shape[0] + stride
        image = cv2.copyMakeBorder(image, stride // 2, stride // 2, stride // 2, stride // 2, cv2.BORDER_CONSTANT, 0)
        png = np.zeros((new_h, new_w), np.uint8)
        image_old = np.zeros((new_h, new_w, 3), np.uint8)

        h_, w_ = image.shape[0], image.shape[1]

        for i in tqdm(range(w_ // stride - 1)):
            for j in range(h_ // stride - 1):
                topleft_x = i * stride
                topleft_y = j * stride
                buttomright_y = topleft_y + target_h
                buttomright_x = topleft_x + target_h
                crop_image = image[topleft_y:buttomright_y, topleft_x:buttomright_x]
                if crop_image.shape[:2] != (target_h, target_h):
                    print(topleft_x, topleft_y, crop_image.shape)

                image_old[topleft_y + dis_wh:buttomright_y - dis_wh, topleft_x + dis_wh:buttomright_x - dis_wh] = \
                    crop_image[dis_wh:target_size[0] - dis_wh, dis_wh:target_size[1] - dis_wh]

                crop_image = tf.cast(tf.expand_dims(crop_image, axis=0), dtype=tf.float32)
                out = tta_interfence(crop_image)
                out = np.squeeze(np.argmax(tf.math.softmax(out), axis=-1))
                # cv2.imwrite(test_mask_path + img[:-4] + "_" + str(i) + str(j) + ".jpg", out * 255)

                png[topleft_y + dis_wh:buttomright_y - dis_wh, topleft_x + dis_wh:buttomright_x - dis_wh] = \
                    out[dis_wh:target_size[0] - dis_wh, dis_wh:target_size[1] - dis_wh]

                x = out[dis_wh:target_size[0] - dis_wh, dis_wh:target_size[1] - dis_wh]

        png = png[hstride:new_h - hstride, hstride:new_w - hstride]
        png = png[:h, :w]
        # png = png * 255
        png = cv2.resize(png, (ww, hh), cv2.INTER_NEAREST)
        cv2.imwrite(test_mask_path + img[:-5] + '_mask.png', png * 255)

        image_old = image_old[hstride:new_h - hstride, hstride:new_w - hstride]
        image_old = image_old[:h, :w]
        image_old = cv2.resize(image_old, (ww, hh))
        cv2.imwrite(test_mask_path + img[:-5] + '.jpg', image_old)

        encs = mask2rle(png)
        new = pd.DataFrame({'id': [img], 'predicted': [encs[0]]}, index=[1])
        test = test.append(new, ignore_index=True)

        # vis_segmentation(np.array(image_old).astype(np.int64), png, ['yin', 'yang'])

    test.to_csv('submission.csv', sep=',', index=False)


if __name__ == '__main__':
    eval_img()
