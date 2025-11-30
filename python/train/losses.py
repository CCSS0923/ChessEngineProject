import torch.nn.functional as F

def loss_policy(pred, target):
    return F.cross_entropy(pred, target)

def loss_value(pred, target):
    return F.mse_loss(pred.squeeze(1), target)
