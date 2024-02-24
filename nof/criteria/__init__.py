from .loss import *


nof_loss = {
    'mse': NOFMSELoss,
    'l1': NOFL1Loss,
    'smoothl1': NOFSmoothL1Loss
}