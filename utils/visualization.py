import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cv2
import pylab
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


if __name__ == '__main__':
    LABEL_NAME = np.asarray(['background', 'nut_center', 'nut_center_n'])
    FULL_LABEL_MAP = np.arange(len(LABEL_NAME)).reshape(len(LABEL_NAME), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    imgfile = '../test_image/Pic_2020_06_10_161901_blockId#42336.bmp'
    pngfile = '../test_image/Pic_2020_06_10_161901_blockId#42336.npy'
    img = cv2.imread(imgfile, 1)
    label = np.load(pngfile)
    img = img[:, :, ::-1]
    # seg_map = cv2.imread(label, 0)
    vis_segmentation(img, label, LABEL_NAME)
    print('Done.')