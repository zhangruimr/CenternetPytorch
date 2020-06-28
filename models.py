import torch as t
from torch import nn
import numpy as np
from sub_modules import *
class CenterNet(nn.Module):
    def __init__(self, heads, num_stacks=2, channels=256, dimension=[256, 384, 384, 384, 512]):
        super(CenterNet, self).__init__()
        self.heads = heads
        self.pre = pre(channels)
        self.channels = channels
        self.dimension = dimension
        self.hourglass = nn.ModuleList()
        for i in range(num_stacks):
            self.hourglass.append(Hourglass(i, heads, channels, dimension))
        self.cb1 = nn.Sequential()
        self.cb1.add_module("cb1_conv", nn.Conv2d(self.channels, self.channels, kernel_size=1, stride= 1, padding=0, bias=False))
        self.cb1.add_module("cb1_bn", nn.BatchNorm2d(self.channels))
        self.cb2 = nn.Sequential()
        self.cb2.add_module("cb2_conv", nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.cb2.add_module("cb2_bn", nn.BatchNorm2d(self.channels))
        self.relu = nn.ReLU()
        self.residual = Residual(self.channels, self.channels)
    def forward(self, inputs):
        x = self.pre(inputs)
        shortcut = x

        x = self.hourglass[0](x)

        x = self.cb1(x)
        shortcut = self.cb2(shortcut)
        x = shortcut + x
        x = self.relu(x)
        x = self.residual(x)

        output = self.hourglass[1](x)
        return output
if __name__ == "__main__":
    x = t.zeros((2, 3, 512, 512))
    model = CenterNet([20, 2, 2])
    x = model(x)
