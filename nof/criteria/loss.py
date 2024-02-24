"""The functions for loss of NOF training
"""

from torch import nn


class NOFLoss(nn.Module):
    def __init__(self):
        super(NOFLoss, self).__init__()
        self.loss = None

    def forward(self, pred, target, valid_mask=None):
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]
        loss = self.loss(pred, target)
        return loss


class NOFMSELoss(NOFLoss):
    """
    MSELoss for predicted scan with real scan
    均方误差(MSE, Mean Squared Error), 即预测值和真实值之差的平方和的平均数
    """

    def __init__(self):
        super(NOFMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')


class NOFL1Loss(NOFLoss):
    """
    L1Loss for predicted scan with real scan
    平均绝对误差(MAE, Mean Absolute Error), 计算方法很简单，取预测值和真实值的绝对误差的平均数即可
    """

    def __init__(self):
        super(NOFL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')


class NOFSmoothL1Loss(NOFLoss):
    """
    SmoothL1Loss for predicted scan with real scan
    """

    def __init__(self):
        super(NOFSmoothL1Loss, self).__init__()
        # 计算分俩方面，当误差在 (-1,1) 上是平方损失，其他情况是L1 损失
        self.loss = nn.SmoothL1Loss(reduction='mean')#对N个样本的loss进行平均之后返回
