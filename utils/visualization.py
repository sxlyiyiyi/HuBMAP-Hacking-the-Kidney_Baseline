import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cv2
import pylab
import os
import os.path as osp
import sys
import numpy as np
from PIL import Image
# 使用系统字体，设置plot显示中文
plt.rcParams['font.sans-serif'] = ['Fangsong']


def create_pascal_label_colormap():
    """
    PASCAL VOC 分割数据集的类别标签颜色映射label colormap
    返回:
        可视化分割结果的颜色映射Colormap
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """
    添加颜色到图片，根据数据集标签的颜色映射 label colormap

    参数:
        label: 整数类型的 2D 数组array, 保存了分割的类别标签 label

    返回:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map, label_names):
    """
    输入图片和分割 mask 的可视化.
    """
    label_names = np.asarray(label_names)
    full_label_map = np.arange(len(label_names)).reshape(len(label_names), 1)
    full_color_map = label_to_color_image(full_label_map)
    plt.figure(figsize=(30, 10))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    # 显示原图
    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title(' input image')

    # 显示分割图
    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    # 显示可视化图
    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.5)
    plt.axis('off')
    plt.title('segmentation overlay')

    def con(c):
        if c == 255:
            return 0
        else:
            return c
    unique_labels = np.unique(seg_map)
    unique_labels = np.unique([con(x) for x in unique_labels])
    ax = plt.subplot(grid_spec[3])
    plt.imshow(full_color_map[unique_labels].astype(np.uint8), interpolation='nearest')
    # y轴的tick从(默认的)左侧移动到右侧
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), label_names[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    # plt.grid('off')
    plt.show()
    # pylab.show()
    plt.pause(0.05)


def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    Examples:
        color_map = get_color_map_list(256)
        pred_mask = PILImage.fromarray(res_map.astype(np.uint8), mode='P')
        pred_mask.putpalette(color_map)
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    return color_map


def gray2pseudo_color(dir_or_file, output_dir):
    """将灰度标注图片转换为伪彩色图片"""
    input = dir_or_file
    output_dir = output_dir
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print('Creating colorful label directory:', output_dir)

    color_map = get_color_map_list(256)
    if os.path.isdir(input):
        for fpath, dirs, fs in os.walk(input):
            for f in fs:
                try:
                    grt_path = osp.join(fpath, f)
                    _output_dir = fpath.replace(input, '')
                    _output_dir = _output_dir.lstrip(os.path.sep)

                    im = Image.open(grt_path)
                    lbl = np.asarray(im)

                    lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
                    lbl_pil.putpalette(color_map)

                    real_dir = osp.join(output_dir, _output_dir)
                    if not osp.exists(real_dir):
                        os.makedirs(real_dir)
                    new_grt_path = osp.join(real_dir, f)

                    lbl_pil.save(new_grt_path)
                    print('New label path:', new_grt_path)
                except:
                    continue
    else:
        print('It\'s not a dir')
