import os
import pandas as pd
import sys
import cv2
import numpy as np
import tifffile as tiff
from tensorflow.keras.models import load_model

IMAGE_SIZE = 384


# 图像转rle编码
def mask2rle(img):
    """
    Efficient implementation of mask2rle, from @paulorzp
    https://www.kaggle.com/xhlulu/efficient-mask2rle
    https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    --
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.pad(pixels, ((1, 1), ))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def my_modelname():
    currentpath = os.path.dirname(sys.argv[0])
    model = load_model(os.path.join(currentpath, 'model.h5'), compile=False)
    return model


# 预测未知图片
# 构建字典
def pred_data(model, path_img_test):
    # 初始化结果文件，定义表头为id,label
    res = ['id,label']
    class_dict = {0: 'neg', 1: 'pos'}
    # 预测图片标签
    for pathi in os.listdir(path_img_test):
        # 合成文件路径
        name_img = os.path.join(path_img_test, pathi)
        # opencv读取图片
        test_image = cv2.imread(name_img)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB) / 255
        test_image = cv2.resize(test_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        # 预测
        test_image = np.expand_dims(test_image, axis=0)
        probs = model.predict(test_image)
        pred_class = np.argmax(probs)
        pred_class = class_dict[pred_class]
        res.append(pathi.split('.')[0] + ',' + str(pred_class))

    tiff_dir = './dataset/hubmap-kidney-segmentation/test'

    test_mask_path = './result/'
    if not os.path.isdir(test_mask_path):
        os.makedirs(test_mask_path)
    target_size = (512, 512)
    stride = 512  # stride 应该大于等于 1/2 target_size, 小于等于target_size
    hstride = stride // 2
    dis_wh = (target_size[0] - stride) // 2
    resize = 0.5

    test = pd.DataFrame(columns=['id', 'predicted'])

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

        png = png[hstride:new_h - hstride, hstride:new_w - hstride]
        png = png[:h, :w]
        # png = png * 255
        png = cv2.resize(png, (ww, hh), cv2.INTER_NEAREST)

        encs = mask2rle(png)
        new = pd.DataFrame({'id': [img], 'predicted': [encs]}, index=[1])
        test = test.append(new, ignore_index=True)

        # vis_segmentation(np.array(image_old).astype(np.int64), png, ['yin', 'yang'])

    test.to_csv('submission.csv', sep=',', index=False)


def main(path_img_test, path_submit):
    # 载入模型
    model = my_modelname()
    # 预测图片
    result = pred_data(model, path_img_test)
    # 写出预测结果
    with open(path_submit, 'w') as f:
        f.write('\n'.join(result) + '\n')


if __name__ == "__main__":
    path_img_test = sys.argv[1]
    path_submit = sys.argv[2]
    main(path_img_test, path_submit)
