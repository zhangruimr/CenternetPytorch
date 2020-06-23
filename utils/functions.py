from __future__ import division
import tensorflow as tf
import torch as t
import math
import numpy as np
import tqdm
import cv2
#(btach, h, w, c)
def gaussian2D(featuremap, sigma = 1):
    np_fm = np.zeros(featuremap.shape).astype("float32")
    indexes = np.where(featuremap == 1)
    indexes = np.stack(indexes, 0).transpose(1, 0)

    for index in indexes:
        batch_i, y, x, cls = int(index[0]), int(index[1]), int(index[2]), int(index[3])
        for i in range(np_fm.shape[1]):
            for j in range(np_fm.shape[2]):
                value = np.exp( -((y - i) * (y - i) + (x - j) * (x - j)) / (2 * sigma * sigma))
                if value > np_fm[batch_i, i, j, cls]:
                    np_fm[batch_i, i, j, cls] = value
    return t.from_numpy(np.transpose(np_fm, (0, 3, 1, 2)))

"""
x = np.zeros((1, 100, 100, 3))
x[0, 50, 50 ,0] = 1
x[0, 10, 10 ,1] = 1
f = gaussian2D(x, 3)
cv2.imshow("win", f[0][:,:,1])
cv2.waitKey(0)
"""
#(btach, h, w, c)
def laebls2featuremap_labels(output, labels, stride, sigma = 2):
   device = t.device("cuda:3" if t.cuda.is_available() else "cpu")

   x_s, y_s = stride


   shape = output.shape
   score_heatmap = np.zeros(shape).astype("float32")
   wh_heatmap = np.zeros((shape[0], shape[1], shape[2], 2)).astype("float32")
   offset_heatmap = np.zeros((shape[0], shape[1], shape[2], 2)).astype("float32")
   pos_mask = np.zeros(shape).astype("float32")
   pos_wh_mask = np.zeros(wh_heatmap.shape).astype("float32")
   #print(labels)
   batch_i, y, x, cls, w , h = labels[:, 0].astype("int32"), \
                               (labels[:, 3] // stride[-1]).astype("int32"),  (labels[:, 2] // stride[0]).astype("int32"), \
                             labels[:, 1].astype("int32"), labels[:, 4], labels[:, 5]
   score_heatmap[batch_i, y, x, cls] = 1

   #cv2.imshow("score",score_heatmap[0][:, :, cls])
   #cv2.waitKey(0)
   pos_mask[batch_i, y, x, cls] = 1
   pos_wh_mask[batch_i, y, x, 0] = 1
   pos_wh_mask[batch_i, y, x, 1] = 1


   offset_heatmap[batch_i, y, x, 0] = y.astype("float32") / y_s - np.floor(y / y_s)
   offset_heatmap[batch_i, y, x, 1] = x.astype("float32") / x_s - np.floor(x / x_s)

   wh_heatmap[batch_i, y, x, 0] = h
   wh_heatmap[batch_i, y, x, 1] = w
   #print("sum", t.sum((t.from_numpy(score_heatmap) > 0).float()), t.sum((t.from_numpy(pos_mask) >= 1).float()))
   gass_mask = gaussian2D(score_heatmap, sigma)

   return t.from_numpy(np.transpose(score_heatmap, (0, 3, 1, 2))).cuda(), t.from_numpy(np.transpose(offset_heatmap, (0, 3, 1, 2))).cuda(), \
          t.from_numpy(np.transpose(wh_heatmap, (0, 3, 1, 2))).cuda(), gass_mask.cuda(), t.from_numpy(np.transpose(pos_mask, (0, 3, 1, 2))).cuda(),\
          t.from_numpy(np.transpose(pos_wh_mask, (0, 3, 1, 2))).cuda()

def heatmap_decoding(heatmap, off, wh, k=100, stride = 4):
    _score, _offset, _size = heatmap, off, wh
    _score = _score[0]
    _offset = _offset[0]
    _size = _size[0]
    #print(_score.argmax(0))
    _score = t.sigmoid(t.from_numpy(_score)).numpy()
    #print(_score[14, :, :])
    #print(_score.shape)


    #cv2.imshow("cat",np.transpose(_score, (1,2,0))[:, :, 11])
    #cv2.imshow("dog",np.transpose(_score, (1,2,0))[:, :, 12])
    #cv2.imshow("w",np.transpose(_size, (1,2,0))[:, :, 1])
    #cv2.waitKey(0)

    detection = []
    mask = np.ones(_score.shape).astype("float32")
    while len(detection) < k:
       index = np.where(_score == np.max(_score))
       #print(index)
       #print(index)
       index = np.concatenate(index, 0)
       #print(index)
       #print("score",_score.shape)
       score = _score[:, index[1], index[2]]
       for i in range(mask.shape[0]):
           mask[i, index[1], index[2]] = 0
       _score = _score * mask
       wh = _size[:, index[1], index[2]]

       off = _offset[:, index[1], index[2]]
       _y, _x, cls = index[1].astype("float32"), index[2].astype("float32"), index[0].astype("float32")
       #print("_y",off.shape)
       _y = _y + off[0]
       _x = _x + off[1]
       y1 = _y * stride - 0.5 * wh[0]
       y2 = _y * stride + 0.5 * wh[0]
       x1 = _x * stride - 0.5 * wh[1]
       x2 = _x * stride + 0.5 * wh[1]
       #print(x1.shape)
       #print(y1.shape)
       #print(x2.shape)
       #print("wh",wh)

       object = np.stack((x1, y1, x2, y2, np.max(score), cls))

       detection.append(object)
       #print(detection.shape)
       #print(len(detection))
    detection = np.stack(detection,0)
    #print("rr",detection)

    return t.from_numpy(detection)
def decode_box(src, detection, size=(512, 512)):
    h, w, c = src.shape
    scale = min(size[0]/w, size[1]/h)
    #print("scale",scale)
    x1, y1, x2, y2 = detection[:, 0], detection[:, 1], detection[:, 2], detection[:, 3]
    if math.ceil(scale * w) == size[0]:
        pad = (512 - math.ceil(h * scale)) // 2
        #print("pad",pad)
        y1 = (y1 - pad) / scale
        y2 = (y2 - pad) / scale
        x1 = x1 / scale
        x2 = x2 / scale
        #print("1")
    elif math.ceil(scale * h) == size[1]:
        pad = (512 - math.ceil(w * scale)) // 2
        #print("pad",x1,x2,y1,y2)
        x1 = (x1 - pad) / scale
        x2 = (x2 - pad) / scale
        y1 = y1 / scale
        y2 = y2 / scale
        #print("2")
        #cv2.waitKey(0)
    else:
        print("error")
        exit()
    detection = t.stack((x1, y1, x2, y2, detection[:, 4], detection[:, 5]), 1)
    return detection
def draw_box(src, detection, classname, color):
    #a = ["cat", "dog"]
    for box in detection:
        #print(box)
        x1 , y1, x2, y2, score, cls = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cls = int(cls)
        #print(x1,y1,x2,y2)
        cv2.rectangle(src, (x1, y1), (x2, y2), color[cls], 2)
        #cv2.putText(src, classname[cls], (x1 , y1+20 ), cv2.FONT_HERSHEY_SIMPLEX, 1, color[cls], 2 )

    return src
def iou(box, detection):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    #print("box",x1,y1,x2,y2)
    _x1, _y1, _x2, _y2 = detection[:,0],detection[:,1],detection[:,2],detection[:,3]
    #print("box2", _x1, _y1, _x2, _y2)
    x_1 = np.maximum(x1,_x1)
    y_1 = np.maximum(y1, _y1)
    x_2 = np.minimum(x2, _x2)
    y_2 = np.minimum(y2, _y2)

    insect = np.clip(x_2 - x_1, a_min=1e-8, a_max=1e10) * np.clip(y_2 - y_1, a_min=1e-8, a_max=1e10)
    size_box = np.clip((y2 - y1), a_min=1e-8, a_max=1e10) * np.clip((x2 - x1), a_min=1e-8, a_max=1e10)
    size_detection = np.clip((_y2 - _y1), a_min=1e-8, a_max=1e10) * np.clip((_x2 - _x1), a_min=1e-8, a_max=1e10)
    union = np.clip(size_box + size_detection - insect, a_min=1e-8, a_max=1e10)

    #print("insect",insect)
    #print("union",union)
    return insect / union

def nms(detection):
    results = []
    score = detection[:, 4]
    seq = np.argsort(-score)
    detection = detection[seq]
    while len(detection) > 0:
        box = detection[0]
        results.append(box)
        ious = iou(box, detection)
        #print("ious",ious)
        #print(i)
        
        #index = ious < 0.75
        index = (ious < 0.7) & (box[5] == detection[:,5])

        #print("indexs", ious < 0.75 )
        detection = detection[index]
        ##print("res", results)
        #print("len", len(detection))
        #print("results", results)

    if len(results) > 0:
       results = np.stack(results, 0)
    else:
       results = np.zeros((0,6))
    return results

def config_obtain(road):
    with open(road) as tp:
        files = tp.read().split("\n")
    config = {}
    for file in  files:
        key = file.split("=")
        if len(key) > 1:
           config[key[0].strip()] = key[1].strip()
    return config
if __name__ == "__main__":
    x = np.zeros((1,800,800,3))
    x[0,50,50,0] = 1
    x[0,100,100,0]=1
    x[0,300,300,0]=1
    x[0,500,500,0]=1
    x = tf.convert_to_tensor(x)
    y = gaussian2D(x, 5)
    cv2.imshow("win", y[0].numpy()[:,:,0])
    cv2.waitKey(0)