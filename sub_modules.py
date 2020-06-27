import torch as t
from torch import nn
import numpy as np
class CBR(nn.Module):
    def __init__(self,pre_channel, channels, kernel_size, stride=1):
        super(CBR,self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv2d = nn.Conv2d(pre_channel, channels, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False)

        self.bn =  nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    def forward(self, inputs):

        x = self.conv2d(inputs)

        x = self.bn(x)
        x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, pre_channels, channels, stride = 1):
        super(Residual, self).__init__()
        self.stride = stride
        self.out_channels = channels

        self.conv2d = nn.Conv2d(pre_channels, channels, kernel_size=3, stride=(stride, stride), padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

        self.conv2d2 = nn.Conv2d(channels, channels, kernel_size=3,  stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv2d3 = nn.Conv2d(pre_channels, channels, kernel_size=1, stride=(stride, stride), padding=0, bias = False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, inputs):
        shortcut = inputs

        in_channels = inputs.shape[-1]
        x = self.conv2d(inputs)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2d2(x)
        x = self.bn2(x)

        if self.stride != 1 or in_channels != self.out_channels:
            shortcut = self.conv2d3(shortcut)
            shortcut = self.bn3(shortcut)

        x = x + shortcut
        x = nn.ReLU()(x)
        return x
class pre(nn.Module):
    def __init__(self, channels):
        super(pre, self).__init__()
        self.cbr = CBR(3, 128, 7, 2)
        self.residual = Residual(128, channels, stride=2)
    def forward(self, inputs):
        x = self.cbr(inputs)

        x = self.residual(x)

        return x

class left_module(nn.Module):
    def __init__(self, pre_channel, dimension=(256, 384, 384, 384, 512)):
        super(left_module, self).__init__()
        self.dimension = dimension
        self.list = nn.ModuleList()
        pre_c = pre_channel
        for i, per_channel in enumerate(self.dimension):
            sub_module = nn.Sequential()
            sub_module.add_module("res_left{}".format(i), Residual(pre_c, per_channel, stride=2))
            pre_c = per_channel
            sub_module.add_module("res2_left{}".format(i), Residual(per_channel, per_channel))
            self.list.append(sub_module)
            sub_module = nn.Sequential()
    def forward(self, inputs):
        outputs = [inputs]
        x = inputs

        for sub in self.list:
            x = sub(x)

            outputs.append(x)

        return outputs

class right_module(nn.Module):
    def __init__(self, pre_channel, dimension=(256, 384, 384, 384, 512)):
        super(right_module, self).__init__()
        self.dimension = dimension
        self.middle_modules = nn.Sequential()
        pre_c  = pre_channel


        for i in range(4):
            self.middle_modules.add_module("res_middle{}".format(i), Residual(pre_c, dimension[-1]))
            pre_c = dimension[-1]

        self.connects = nn.ModuleList()
        self.rights = nn.ModuleList()
        for i in reversed(range(len(dimension))):
            sequence = nn.Sequential()
            sequence.add_module("res1_top{}".format(i), Residual(dimension[max(i-1, 0)], dimension[max(i-1, 0)]))

            pre_c2 = dimension[max(i-1, 0)]#(256, 384, 384, 384, 512))

            sequence.add_module("res2_top{}".format(i), Residual(dimension[max(i-1, 0)], dimension[max(i-1, 0)]))
            self.connects.append(sequence)

            sequence = nn.Sequential()

            right_seq = nn.Sequential()
            right_seq.add_module("res_blow{}".format(i), Residual(pre_c, dimension[i]))

            pre_c = dimension[i]

            right_seq.add_module("res4{}".format(i), Residual(pre_c, dimension[max(i-1, 0)]))
            pre_c = dimension[max(i-1, 0)]


            right_seq.add_module("up{}".format(i), nn.ConvTranspose2d(dimension[max(i-1, 0)],dimension[max(i-1, 0)], kernel_size=2, stride=2, bias=True,padding=0, output_padding=0))
            self.rights.append(right_seq)

            right_seq = nn.Sequential()

    def forward(self, inputs):
        x = self.middle_modules(inputs[-1])
        #short = inputs[-1]
        inputs.pop(-1)
        for i in range(len(self.connects)):


            shortcut = self.connects[i](inputs[-1])

            inputs.pop(-1)

            x = self.rights[i](x)

            x = shortcut + x

        return x

class Hourglass(nn.Module):
    def __init__(self, i, heads, pre_channels, dimension=[256, 384, 384, 384, 512]):
        super(Hourglass, self).__init__()
        self.heads = heads
        self.pre_channels = pre_channels
        self.dimension = dimension
        self.left = left_module(pre_channels, self.dimension)

        self.right = right_module(self.dimension[-1], self.dimension)

        self.cbr = CBR(self.dimension[0],256,3)
        self.num = i
        if i > 0:
          self.creatHead = nn.ModuleList()
          for j in range(len(heads)):
            h = nn.Sequential()
            h.add_module("head{}_conv2d{}".format(i,j), nn.Conv2d(256, 256,  kernel_size=3, stride=1, bias=True, padding=1))
            h.add_module("head{}_relu{}".format(i,j), nn.ReLU())
            h.add_module("head{}_conv2d2{}".format(i,j), nn.Conv2d(256, heads[j], kernel_size=1, stride=1, padding=0, bias=True))
            self.creatHead.append(h)
    def forward(self, inputs):
        output = []
        x = self.left(inputs)

        x = self.right(x)

        x = self.cbr(x)
        if self.num > 0:
          for i in range(len(self.creatHead)):

              output.append(self.creatHead[i](x))
          return output
        else:
            return x

if __name__ == "__main__":
    x = t.zeros((2, 256, 128, 128))
    model = Hourglass(0,[20, 2, 2], 256)
    output, x = model(x)
