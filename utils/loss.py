import tensorflow as tf
import timeit
from tensorflow.keras import backend as K


# 交叉熵损失函数
def CE_Loss(class_nums, ignore=False):
    def ce_loss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, class_nums])
        y_true = tf.cast(tf.reshape(y_true, [-1, ]), tf.int32)
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, class_nums - 1)), -1)
            y_pred = tf.gather(y_pred, indices)
            y_true = tf.gather(y_true, indices)

        loss = tf.reduce_mean(
            tf.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred))

        return loss
    return ce_loss


# 带权重的交叉熵损失，可以平衡原本数量
def W_CELoss(num_classes, weights=None, ignore=False):
    def w_celoss(y_true, y_pred):
        """
        :param weight:
        :param y_pred:神经网络输出层的输出, 未进行缩放的原始值 [batchsize, height, width, num_classes]
        :param y_true:为样本的真实标签，每一个值∈[0，num_classes)，其实就是代表了batch中对应样本的类别[batchsize, height, width]
        :return:
        """
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        y_true = tf.cast(tf.reshape(y_true, [-1, ]), tf.int32)
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, num_classes - 1)), 1)
            y_pred = tf.gather(y_pred, indices)
            # labels = tf.cast(tf.expand_dims(tf.gather(labels, indices), axis=-1), tf.float32)
            y_true = tf.gather(y_true, indices)
        if weights is None:
            cross_entropy = tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred))

        elif isinstance(weights, (float, int)):
            cross_entropy = weights * tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred))

        elif isinstance(weights, list):
            class_weights = tf.constant(weights)
            class_weights = tf.gather(class_weights, y_true)
            cross_entropy = tf.reduce_mean(
                weights * tf.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred))

        else:
            raise ValueError('unsuportted weights')

            # class_weights = tf.constant(weight)
            # class_weights = tf.reduce_sum(class_weights * tf.one_hot(labels, logits.shape[3]), axis=-1)
            # unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, logits.shape[3]), logits)
            # weighted_losses = unweighted_losses * class_weights
            # loss = tf.reduce_mean(weighted_losses)

        return cross_entropy
    return w_celoss


# 用于多分类（包括二分类）的Dice Loss
def DiceLoss(num_classes, loss_type='jaccard', smooth=1e-5, ignore=False):
    def diceloss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        y_true = tf.reshape(y_true, [-1, ])
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, num_classes - 1)), 1)
            y_pred = tf.gather(y_pred, indices)
            # labels = tf.cast(tf.expand_dims(tf.gather(labels, indices), axis=-1), tf.float32)
            y_true = tf.cast(tf.gather(y_true, indices), tf.int32)

        y_yrue = tf.cast(tf.one_hot(y_true, depth=num_classes, on_value=1, off_value=0), tf.float32)

        inse = tf.reduce_sum(y_pred * y_yrue, axis=0)
        if loss_type == 'jaccard':
            x = tf.reduce_sum(y_pred * y_pred, axis=0)
            y = tf.reduce_sum(y_yrue * y_yrue, axis=0)
        elif loss_type == 'sorensen':
            x = tf.reduce_sum(y_pred, axis=0)
            y = tf.reduce_sum(y_yrue, axis=0)
        else:
            raise Exception("Unkown loss_type")

        dice1 = (2. * inse + smooth) / (x + y + smooth)
        dice1 = tf.reduce_mean(dice1)

        return 1. - dice1
    return diceloss


# 用于多分类的Dice Loss和多元交叉熵Loss加权
def CE_Dice_Loss(num_classes, loss_type='jaccard', smooth=1e-5, ignore=False):
    def cediceloss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        y_true = tf.cast(tf.reshape(y_true, [-1, ]), tf.int32)
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, num_classes - 1)), axis=-1)
            logits = tf.gather(y_pred, indices)
            # labels = tf.cast(tf.expand_dims(tf.gather(labels, indices), axis=-1), tf.float32)
            y_true = tf.gather(y_true, indices)

        ce_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred))

        y_true = tf.cast(tf.one_hot(y_true, depth=num_classes, on_value=1, off_value=0), tf.float32)

        inse = tf.reduce_sum(y_pred * y_true, axis=0)
        if loss_type == 'jaccard':
            x = tf.reduce_sum(y_pred * y_pred, axis=0)
            y = tf.reduce_sum(y_true * y_true, axis=0)
        elif loss_type == 'sorensen':
            x = tf.reduce_sum(y_pred, axis=0)
            y = tf.reduce_sum(y_true, axis=0)
        else:
            raise Exception("Unkown loss_type")

        dice1 = (2. * inse + smooth) / (x + y + smooth)
        dice1 = 1. - tf.reduce_mean(dice1)

        return dice1 + ce_loss
    return cediceloss


# 用于多分类（包括二分类）的Focal Loss
def FocalLoss(num_classes, alpha=0.25, gamma=2, label_smoothing=0.05, ignore=False):
    def focalloss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        y_true = tf.cast(tf.reshape(y_true, [-1, ]), tf.int32)
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, num_classes - 1)), 1)
            y_pred = tf.gather(y_pred, indices)
            y_true = tf.gather(y_true, indices)

        y_true = tf.cast(tf.one_hot(y_true, num_classes), tf.float32)

        # compute focal loss multipliers before label smoothing, such that it will
        # not blow up the loss.
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = (1.0 - p_t) ** gamma

        # apply label smoothing for cross_entropy for each entry.
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        ce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)

        # compute the final loss and return
        loss = tf.reduce_mean(alpha_factor * modulating_factor * ce)
        return loss
    return focalloss


# 用于二分类的Focal Loss
def B_FocalLoss(alpha=0.25, gamma=2, ignore=False):
    def focalloss(y_true, y_pred):
        # # labels = tf.cast(tf.squeeze(labels), dtype=tf.int32)
        # # class_alpha = tf.constant(alpha)
        # # class_alpha = tf.reduce_sum(class_alpha * tf.one_hot(labels, logits.shape[3]), axis=-1)
        # # unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, logits.shape[3]), logits)
        # # weighted_losses = unweighted_losses * class_alpha * ((1 - tf.math.softmax(logits))**gamma)
        # # loss = tf.reduce_mean(weighted_losses)
        # # 二分类
        # # labels = tf.cast(tf.squeeze(labels), dtype=tf.float32)
        y_pred = tf.reshape(y_pred, [-1, 2])
        y_true = tf.cast(tf.reshape(y_true, [-1, ]), tf.int32)
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, 1)), 1)
            y_pred = tf.gather(y_pred, indices)
            y_true = tf.gather(y_true, indices)
        y_true = tf.cast(tf.one_hot(y_true, 2), tf.float32)

        y_pred = tf.sigmoid(y_pred)
        pos_loss = -alpha * tf.pow(1. - y_pred, gamma) * tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) * y_true
        neg_loss = -(1-alpha) * tf.pow(y_pred, gamma) * tf.math.log(tf.clip_by_value(1. - y_pred, 1e-8, 1.0)) * (1. - y_true)
        loss = tf.reduce_mean(pos_loss + neg_loss)
        return loss
    return focalloss


#  用于多分类（包括二分类）的Tversky Loss
def Tversky_Loss(num_classes, alpha=0.7, smooth=1e-8, ignore=False):
    def tversky_loss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        y_true = tf.reshape(y_true, [-1, ])
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, num_classes - 1)), 1)
            y_pred = tf.gather(y_pred, indices)
            # labels = tf.cast(tf.expand_dims(tf.gather(labels, indices), axis=-1), tf.float32)
            y_true = tf.cast(tf.gather(y_true, indices), tf.int32)

        target = tf.cast(tf.one_hot(y_true, depth=num_classes, on_value=1, off_value=0), tf.float32)
        tp = tf.reduce_sum(target * y_pred, axis=0)
        fn = tf.reduce_sum(target * (1 - y_pred), axis=0)
        fp = tf.reduce_sum((1 - target) * y_pred, axis=0)

        tversky = (tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth)
        tversky = tf.reduce_mean(tversky)
        loss = 1. - tversky

        return loss
    return tversky_loss


# 用于多分类（包含二分类）的Focal Loss和Tversky_loss加权
def Focal_Tversky_Loss(num_classes, alpha=0.7, smooth=1e-8, gamma=0.75, ignore=False):
    def focal_tversky_loss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        y_true = tf.reshape(y_true, [-1, ])
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, num_classes - 1)), 1)
            y_pred = tf.gather(y_pred, indices)
            # labels = tf.cast(tf.expand_dims(tf.gather(labels, indices), axis=-1), tf.float32)
            y_true = tf.cast(tf.gather(y_true, indices), tf.int32)

        y_true = tf.cast(tf.one_hot(y_true, depth=num_classes, on_value=1, off_value=0), tf.float32)
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)

        tversky = (tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth)
        focaltversky = tf.reduce_mean(tf.pow((1. - tversky), gamma))

        return focaltversky
    return focal_tversky_loss


# 改善Dice Loss，将多个类别的Dice Loss进行整合，使用一个参数作为分割结果的量化指标
def Generalized_Dice_Loss(num_classes, ignore=False):
    def generalized_dice(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        y_true = tf.cast(tf.reshape(y_true, [-1, ]), tf.int32)
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, num_classes - 1)), 1)
            y_pred = tf.gather(y_pred, indices)
            # labels = tf.cast(tf.expand_dims(tf.gather(labels, indices), axis=-1), tf.float32)
            y_true = tf.gather(y_true, indices)

        y_true = tf.cast(tf.one_hot(y_true, depth=num_classes, on_value=1, off_value=0), tf.float32)
        w = tf.reduce_sum(y_true, axis=0)
        w = 1. / (w**2 + 1e-8)
        # Compute gen dice coef:
        numerator = w * tf.reduce_sum(y_pred * y_true, axis=0)
        denominator = w * tf.reduce_sum(y_pred + y_true, axis=0)

        gen_dice_coef = 2. * tf.reduce_sum(numerator) / tf.reduce_sum(denominator)

        return 1. - gen_dice_coef
    return generalized_dice


# OHEN交叉熵损失函数
def OHEM_CE_Loss(num_classes, top_k, ignore=False):
    """
    :param ignore:
    :param top_k:
    :param num_classes:
    # :param thresh: 损失函数阈值, 0.65
    # :param n_min: MIN_SAMPLE_NUMS: 262144    1:8
    :return:
    """
    def ohemce_loss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        y_true = tf.cast(tf.reshape(y_true, [-1, ]), tf.int32)
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, num_classes - 1)), 1)
            y_pred = tf.gather(y_pred, indices)
            y_true = tf.gather(y_true, indices)
        # 计算cross entropy loss
        loss = tf.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)
        loss, _ = tf.nn.top_k(loss, tf.size(loss), sorted=True)

        # apply ohem
        # ohem_thresh = tf.multiply(-1.0, tf.math.log(thresh))
        # ohem_cond = tf.greater(loss[n_min], ohem_thresh)
        # loss_select = tf.cond(pred=ohem_cond,
        #                       true_fn=lambda: tf.gather(loss, tf.squeeze(tf.where(tf.greater(loss, ohem_thresh)), 1)),
        #                       false_fn=lambda: loss[:n_min])
        loss_select = loss[:top_k]
        loss_value = tf.reduce_mean(loss_select)

        return loss_value
    return ohemce_loss


# Jaccard/IoU 损失函数
def Jaccard_Loss(num_classes, smooth=0.05, ignore=False):
    def jaccard_loss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        y_true = tf.cast(tf.reshape(y_true, [-1, ]), tf.int32)
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, num_classes - 1)), 1)
            y_pred = tf.gather(y_pred, indices)
            y_true = tf.gather(y_true, indices)
        y_true = tf.cast(tf.one_hot(y_true, depth=num_classes, on_value=1, off_value=0), tf.float32)

        inse = tf.reduce_sum(y_pred * y_true, axis=0)
        total = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)
        union = total - inse

        IoU = (inse + smooth) / (union + smooth)

        return 1. - tf.reduce_mean(IoU)
    return jaccard_loss


# Jaccard/IoU 损失函数和多元交叉熵Loss加权
def CE_Jaccard_Loss(num_classes, smooth=0.05, ignore=False):
    def ce_jaccard_loss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        y_true = tf.cast(tf.reshape(y_true, [-1, ]), tf.int32)
        if ignore is True:
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, num_classes - 1)), 1)
            y_pred = tf.gather(y_pred, indices)
            y_true = tf.gather(y_true, indices)

        ce_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred))
        y_true = tf.cast(tf.one_hot(y_true, depth=num_classes, on_value=1, off_value=0), tf.float32)

        inse = tf.reduce_sum(y_pred * y_true, axis=0)
        total = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)
        union = total - inse

        IoU = (inse + smooth) / (union + smooth)

        return 1. - tf.reduce_mean(IoU) + ce_loss
    return ce_jaccard_loss


# l2正则项损失函数
def L2_REG_Loss(var_list, weights_decay):
    def l2reg_loss():
        l2_reg_loss = tf.constant(0.0, tf.float32)
        for vv in var_list:
            if 'beta' in vv.name or 'gamma' in vv.name or 'b:0' in vv.name.split('/')[-1]:
                continue
            else:
                l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
        l2_reg_loss *= weights_decay
        l2_reg_loss = tf.identity(l2_reg_loss, 'l2_loss')

        return l2_reg_loss
    return l2reg_loss


def get_loss(cfg, loss_type):
    """
        loss_type: 'wce_loss', 'ce_loss', 'dice_loss', 'OHEM_loss', 'focal_loss', 'cedice_loss', 'tversky_loss',
                   'focal_tversky_loss', 'generalized_dice_loss', 'jaccard_loss', 'ce_jaccard_loss'
        classes:
        ignore:
        label_smoothing:
        weights:
        top_k:
        alpha:
        gamma:

    """
    classes = cfg.DATASET.N_CLASSES
    weights = cfg.LOSS.WEIGHTS
    top_k = cfg.LOSS.TOPK
    alpha = cfg.LOSS.ALPHA
    gamma = cfg.LOSS.GAMMA
    label_smoothing = cfg.LOSS.LABEL_SMOOTHING
    ignore = cfg.LOSS.IGNORE

    if loss_type == 'wce_loss':
        loss = W_CELoss(classes, weights, ignore=ignore)
    elif loss_type == 'ce_loss':
        loss = CE_Loss(classes, ignore=ignore)
    elif loss_type == 'dice_loss':
        loss = DiceLoss(classes, ignore=ignore)
    elif loss_type == 'OHEM_loss':
        loss = OHEM_CE_Loss(classes, top_k, ignore)
    elif loss_type == 'focal_loss':
        loss = FocalLoss(num_classes=classes, alpha=alpha, gamma=gamma,
                         label_smoothing=label_smoothing, ignore=ignore)
    elif loss_type == 'cedice_loss':
        loss = CE_Dice_Loss(classes, ignore=ignore)
    elif loss_type == 'tversky_loss':
        loss = Tversky_Loss(classes, ignore=ignore)
    elif loss_type == 'focal_tversky_loss':
        loss = Focal_Tversky_Loss(classes, alpha=alpha, gamma=gamma, ignore=ignore)
    elif loss_type == 'generalized_dice_loss':
        loss = Generalized_Dice_Loss(classes, ignore=ignore)
    elif loss_type == 'jaccard_loss':
        loss = Jaccard_Loss(classes, ignore=ignore)
    elif loss_type == 'ce_jaccard_loss':
        loss = CE_Jaccard_Loss(classes, ignore=ignore)
    else:
        raise TypeError('Unsupported loss type')

    return loss
