from __future__ import division
import torch as t
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
import numpy as np
from PIL import Image
import math
import random
import cv2
def train_process(img, label, size, flip=True):
    w, h = size
    _c, _h, _w,= img.shape
    scale = min(h / _h, w / _w)
    img = F.interpolate(img.unsqueeze(0), (math.ceil(scale * _h), math.ceil(scale * _w)), mode="nearest").squeeze()
    label = label
    if math.ceil(scale * _h) == h:
        padding = (w - math.ceil(scale * _w)) // 2
        pad = (padding, w - math.ceil(scale * _w) -  padding, 0, 0)

        y = label[:, 2]
        h2 = label[:, 4]
        x = (_w * scale * label[:, 1] + pad[0]) / w
        w2 = label[:, 3] * _w * scale / w
    else:
        padding = (h - math.ceil(scale * _h)) // 2
        pad = (0, 0, padding, h - math.ceil(scale * _h) - padding)

        x = label[:, 1]
        w2 = label[:, 3]
        y = (_h * scale * label[:, 2] + pad[2]) / h
        h2 = label[:, 4] * _h * scale / h
    img = F.pad(img.unsqueeze(0), pad).squeeze(0)

    label[:, 1] = x
    label[:, 2] = y
    label[:, 3] = w2
    label[:, 4] = h2
    if flip:
        if random.randint(0, 1) >= 0.5:
            img = t.flip(img, [2])
            label[:, 1] = 1 - label[:, 1]
    label[:, 1] = label[:, 1] * w
    label[:, 2] = label[:, 2] * h
    label[:, 3] = label[:, 3] * w
    label[:, 4] = label[:, 4] * h
    labels = np.zeros((label.shape[0], 6))
    labels[:, 1:] = label
    labels = t.from_numpy(labels).float()
    return img, labels
def test_process(img, size):
    w, h = size
    _c, _h, _w,= img.shape
    scale = min(h / _h, w / _w)
    img = F.interpolate(img.unsqueeze(0), (math.ceil(scale * _h), math.ceil(scale * _w)), mode="nearest").squeeze()
    if math.ceil(scale * _h) == h:
        padding = (w - math.ceil(scale * _w)) // 2
        pad = (padding, w - math.ceil(scale * _w) -  padding, 0, 0)
    else:
        padding = (h - math.ceil(scale * _h)) // 2
        pad = (0, 0, padding, h - math.ceil(scale * _h) - padding)

    img = F.pad(img.unsqueeze(0), pad).squeeze(0)

    return img
class TrainDataset(Dataset):
    def __init__(self, img_road, size):
        with open(img_road, 'r') as tp:
            self.imgs_road = tp.read()
            self.imgs_road = [i for i in self.imgs_road.split("\n") if i != '']
        self.labels_road = [i.replace("JPEGImages", "labels").replace("jpg", "txt"). \
                                replace("jpeg", "txt").replace("png", "txt") for i in self.imgs_road]
        self.size = size
    def __getitem__(self, index):
        img = Image.open(self.imgs_road[index])
        label = np.loadtxt(self.labels_road[index]).astype("float32")
        if len(label.shape) < 2:
            if label.shape[0] == 0:
                label = np.zeros((0, 5))
            else  :
                label = np.expand_dims(label, 0)
        img = T.ToTensor()(img)

        if len(img.shape) < 3:
            print("image error, road in {}".format(self.imgs_road[index]))
        img, label = train_process(img, label, self.size)

        return img, label, self.imgs_road[index]
    def collate_fn(self, batch):

        imgs, labels, roads = list(zip(*batch))
        for i in range(len(labels)):

            labels[i].detach()[:,0] = i
        imgs = t.stack(imgs, 0)
        labels = t.cat(labels, 0)
        return imgs, labels, roads
    def __len__(self):
        return len(self.imgs_road)

class TestDataset(Dataset):
    def __init__(self, img_road, size):
        with open(img_road, 'r') as tp:
            self.imgs_road = tp.read()
            self.imgs_road = [i for i in self.imgs_road.split("\n") if i != '']
        self.size = size
    def __getitem__(self, index):
        img = Image.open(self.imgs_road[index])
        img = T.ToTensor()(img)

        if len(img.shape) < 3:
            print("image error, road in {}".format(self.imgs_road[index]))
        img = test_process(img, self.size)
        return img, self.imgs_road[index]

    def collate_fn(self, batch):

        imgs, roads = list(zip(*batch))
        imgs = t.stack(imgs, 0)
        return imgs,  roads
    def __len__(self):
        return len(self.imgs_road)

if __name__ == "__main__":
     dataset  = TrainDataset("./data/train.txt", (512, 512))
     dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
     for img, labels,  roads in dataloader:
         print(labels)
         imgs = img[0].permute(1,2,0).contiguous().numpy()[:,:,::-1].copy()
         box = labels[1].numpy()
         x, y, w, h = int(box[2]), int(box[3]), int(box[4]), int(box[5])

         cv2.rectangle(imgs, (int(x-0.5*w), int(y-0.5*h)), (int(x+0.5 * w), int(y + 0.5 * h)), (0, 0, 255), 2, 1)
         cv2.imshow("win", imgs)
         cv2.waitKey(0)