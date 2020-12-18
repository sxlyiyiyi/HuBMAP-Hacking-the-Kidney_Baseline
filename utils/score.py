import numpy as np
import tensorflow as tf


# 混淆矩阵
def fast_hist(a, b, n):
    """
    产生n×n的混淆矩阵, 横看真实, 竖看预测
    :param a: 转化成一维数组的标注图，形状(H×W,)
    :param b: 转化成一维数组的预测图，形状(H×W,)
    :param n: 类别数
    :return:
    """
    # k为掩膜（去除了255这些点(即标签图中的白色轮廓)
    k = (a >= 0) & (a < n)

    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


class SegmentationMetric(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusionMatrix = np.zeros((self.num_class,) * 2)

    def pixelAccuracy(self):
        """
        PA = acc = (TP + TN) / (TP + TN + FP + TN)
        :return: all class overall pixel accuracy
        """
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        """
        acc = (TP) / TP + FP
        :return: each category pixel accuracy(A more accurate way to call it precision)
        """
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表， 表示各类别的准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def mIoU(self):
        """
        Intersection = TP Union = TP + FP + FN
        IoU = TP / (TP + FP + FN)
        :return:
        """
        intersection = np.diag(self.confusionMatrix)  # 取对角元素, 返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)

        return mIoU

    def IoU(self, label_id):
        # interseection = self.confusionMatrix[label_id, label_id]
        # union = np.sum(self.confusionMatrix[label_id, :]) + np.sum(self.confusionMatrix[:, label_id]) - interseection
        # IoU = interseection/union
        intersection = np.diag(self.confusionMatrix)  # 取对角元素, 返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union

        return IoU[label_id]

    def genConfusionMatrix(self, img_prdict, img_label):
        """
        confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
        L/P     P     N
        P      TP    FP
        N      FN    TN
        :param img_prdict:
        :param img_label:
        :return:
        """
        mask = (img_label >= 0) & (img_label < self.num_class)
        label = self.num_class * img_label[mask] + img_prdict[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusionMatrix = count.reshape(self.num_class, self.num_class)
        return confusionMatrix

    def FWIoU(self):
        """
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        :return:
        """
        freq = np.sum(self.confusionMatrix, axis=1) / (np.sum(self.confusionMatrix) + 1e-8)
        iu = np.diag(self.confusionMatrix) / (np.sum(self.confusionMatrix, axis=1) +
                                              np.sum(self.confusionMatrix, axis=0) -
                                              np.diag(self.confusionMatrix) + 1e-8)
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()

        return fwiou

    def dice(self):
        """
        2TP / (FP + 2TP + FN)
        :return:
        """
        up = 2. * np.diag(self.confusionMatrix)
        down = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0)
        dice = np.nanmean(up / down)
        return dice

    def addBatch(self, img_label, img_predict):
        assert img_predict.shape == img_label.shape
        self.confusionMatrix += self.genConfusionMatrix(img_predict, img_label)

    def reset(self):
        self.confusionMatrix = np.zeros((self.num_class, self.num_class))


# 计算参数量
def param_count():
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


# 计算FLOPs
def count_flops(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))


if __name__ == '__main__':
    img_pre = np.array([0, 0, 1, 1, 2, 2])
    img_lab = np.array([0, 1, 1, 1, 2, 2])
    aa = fast_hist(img_lab, img_pre, 3)
    metric = SegmentationMetric(3)
    metric.addBatch(img_lab, img_pre)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    miou = metric.mIoU()
    print('pa is : %f' % pa)
    print('cpa is :')  # 列表
    print(cpa)
    print('mpa is : %f' % mpa)
    print('mIoU is : %f' % miou)
