import torch as t
import numpy as np
import cv2

class FocalLoss(t.nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    def forward(self, output, label, gass_mask, pos_mask, a=2, b=4):

        output = output.float()
        output = t.sigmoid(output)
        label = label.float()
        gass_mask = gass_mask.float()
        pos_mask = pos_mask.float()
        pos_loss = t.pow(1 - output, a) * t.log(t.clamp(output, min=1e-5, max=1 - 1e-5)) * pos_mask

        neg_loss = t.pow(1 - gass_mask, b) * t.pow(output, a) * t.log(t.clamp(1 - output, min=1e-5, max=1 - 1e-5)) * ( \
                    1 - pos_mask)
        loss = t.sum(pos_loss + neg_loss)
        num = t.sum(pos_mask)
        print("objectNum:", num)
        loss = -(loss / num)

        return loss
class L1loss(t.nn.Module):
    def __init__(self):
        super(L1loss, self).__init__()
    def forward(self, output, label, pos_mask):

        output = output.float()
        label = label.float()
        pos_mask = pos_mask.float()
        loss = t.sum(t.abs(output - label) * pos_mask)
        num = t.sum(pos_mask)
        return loss / num
class wholeLoss(t.nn.Module):
    def __init__(self):
        super(wholeLoss, self).__init__()
    def forward(self, focalloss, sizeloss, offloss, weightsize=0.1, weightoff=1):
        return focalloss + weightsize * sizeloss + weightoff * offloss
