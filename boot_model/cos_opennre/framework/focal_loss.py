from torch import nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pred, target):
        pred = self.softmax(pred)
        one_hot_target = torch.zeros(target.size(0), self.class_num).cuda()
        one_hot_target.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.alpha is not None:
            self.alpha = torch.tensor(self.alpha).cuda()
            assert len(self.alpha) == self.class_num
            batch_loss = - self.alpha * torch.pow(1-pred, self.gamma) * pred.log() * one_hot_target
        else:
            batch_loss = - torch.pow(1-pred, self.gamma) * pred.log() * one_hot_target

        batch_loss = batch_loss.sum(dim=1)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

