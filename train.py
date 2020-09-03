import torch as t
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from models import *
from utils.functions import *
from utils.losses import *
from datasets import *


def init(model):
    for name, params in model.named_parameters():
        if name.find("bias") > 0:
            nn.init.constant_(params, 0)
        elif name.find("weight") > 0:
            if name.find("conv2d") > 0 or name.find("up") > 0:
                nn.init.normal_(params, 0, 0.01)
            else:
                nn.init.constant_(params, 1)


def train(**config):
    trainroad = config["trainroad"]
    input_w, input_h = (int(config["input_w"]), int(config["input_h"]))
    classes = int(config["class"])
    batch = int(config["batch"])
    epoches = int(config["epoch"])
    head = [int(i.strip()) for i in config["head"].split(",")]
    weight_load = config["weight_load"]
    print(weight_load)
    lr = float(config["lr"])
    os.makedirs(name="weights", exist_ok=True)
    if t.cuda.is_available():
        print("GPU Training")
    else:
        print("CPU Training")
    device = t.device("cuda:3" if t.cuda.is_available() else "cpu")
    dataset = TrainDataset(trainroad, (input_w, input_h))

    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True, collate_fn=dataset.collate_fn)
    net = CenterNet([classes, 2, 2])

    if weight_load != "":
        print("----加载模型参数----")
        net.load_state_dict(t.load(weight_load))
    else:
        print("----初始化模型----")
        init(net)
    for name, para in model.named_parameters():
        print(name, para)
    net.train()
    net = net.cuda()

    focal_loss = FocalLoss().cuda()
    size_loss = L1loss().cuda()
    off_loss = L1loss().cuda()
    whole_loss = wholeLoss().cuda()

    optimizer = t.optim.Adam(net.parameters(), lr=lr)
    #optimizer = t.optim.SGD(net.parameters(), lr=lr)

    scheduler = t.optim.lr_scheduler.StepLR(optimizer, 100, 0.1)

    for epoch in range(epoches):

        print("Epoch---{}".format(epoch))
        for i, (imgs, labels, road) in enumerate(dataloader):

            print("-----Epoch-{}-----Batch-{}".format(epoch, i).format(epoch))
            loader_num = len(dataloader)

            imgs = imgs.cuda()
            labels = labels.cuda()

            output = net(imgs)

            print("max", t.max(output[0]))
            print("min", t.min(output[0]))
            print("number>0", t.sum(output[0] >= 0))
            print("number<0", t.sum(output[0] < 0))
            score_heatmap, off_heatmap, size_heatmap, gass_mask, pos_mask, pos_wh_mask \
                = laebls2featuremap_labels(np.transpose(output[0].cpu().detach().numpy(), (0, 2, 3, 1)),
                                           labels.cpu().detach().numpy(), (4, 4), 1)
            focalloss = focal_loss(output[0], score_heatmap, gass_mask, pos_mask)


            print("focalloss:", focalloss)
            offloss = off_loss(output[1], off_heatmap, pos_wh_mask)
            print("offloss:", offloss)
            sizeloss = size_loss(output[2], size_heatmap, pos_wh_mask)
            print("sizeloss:", sizeloss)
            loss = whole_loss(focalloss, sizeloss, offloss)
            print("loss:", loss)

            loss.backward()
            if (i + 1) % 16 ==0:
               optimizer.step()
               optimizer.zero_grad()


        scheduler.step()
        if (epoch + 1) % 8 == 0:
            if epoch < epoches - 1:
                t.save(net.state_dict(), "weights/Epoch{}.pth".format(epoch))
            else:
                t.save(net.state_dict(), "weights/best.pth")


if __name__ == "__main__":
    config = config_obtain("config.cfg")
    train(**config)
