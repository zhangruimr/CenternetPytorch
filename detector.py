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
def detector(**config):
    trainroad = config["trainroad"]
    input_w, input_h = (int(config["input_w"]), int(config["input_h"]))
    testroad = config["testroad"]
    resultsroad = config["resultsroad"]
    weightsroad = config["weightsroad"]
    classes = int(config["class"])
    classname = config["classname"].split(",")
    head = [int(i.strip()) for i in config["head"].split(",")]
    os.makedirs(resultsroad, exist_ok=True)
    model = CenterNet(head)
    model = t.nn.DataParallel(model, device_ids=[0,1])
    model = model.cuda()
    model.eval()
    if os.path.exists(weightsroad):
        model.load_state_dict(t.load(weightsroad))
    else:
        print("have no file : {}".format(weightsroad))
        print("please change the term named weightsroad in file config.cfg")
        exit()
    test_dataset = TestDataset(testroad, (input_w, input_h))
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=test_dataset.collate_fn)
    color = []
    for i in range(len(classname)):
        color .append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    for img, road in tqdm.tqdm(testloader):
        #cv2.imshow("SRC", np.transpose(img[0].numpy(), (1,2,0)))
        #print(road)
        img = img.cuda()
        with t.no_grad():
            heatmap = model(img)
        detection = heatmap_decoding(heatmap[0].cpu().detach().numpy(), heatmap[1].cpu().detach().numpy(), heatmap[2].cpu().detach().numpy())
        detection = detection[detection[:, 4] > 0.5]
        # print(detection)

        src = cv2.imread(road[0])
        detection = decode_box(src, detection, (input_w, input_h))
        detection = nms(detection)
        #print("detect",detection)
        src = draw_box(src, detection, classname, color)

        cv2.imshow("results", src)
        cv2.waitKey(1)
        cv2.imwrite(resultsroad + road[0].split("/")[-1], src)
        i+=1
        if i > 200 :
            break


if __name__ == "__main__":
    config = config_obtain("config.cfg")
    detector(**config)