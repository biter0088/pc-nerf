import torch
from .pointcloud_metrics import eval_pts


def abs_error(pred, gt, valid_mask=None):
    value = torch.abs(pred - gt)
    if valid_mask is not None:
        value = value[valid_mask]

    return torch.mean(value)#


def acc_thres(pred, gt, valid_mask=None):
    error = torch.abs(pred - gt)
    if valid_mask is not None:
        error = error[valid_mask]
    # acc_thres = error < 2
    acc_thres = error < 0.2 #    
    # acc_thres = error < 0.5 #        
    acc_thres = torch.sum(acc_thres) / acc_thres.shape[0] * 100

    return acc_thres


def eval_points(pred_pts, gt_pts, valid_mask=None):
    if valid_mask is not None:
        pred_pts = pred_pts[valid_mask].cpu().numpy()
        gt_pts = gt_pts[valid_mask].cpu().numpy()

    cd, fscore = eval_pts(pred_pts, gt_pts)

    return cd, fscore
