import torch as t
from datasets import *
from models import *
import os

from utils.functions import *
def video_detect(**config):

    input_w, input_h = (int(config["input_w"]), int(config["input_h"]))
    videoroad = config["videoroad"]
    resultsroad = config["resultsroad"]
    weightsroad = config["weightsroad"]
    classes = int(config["class"])
    classname = config["classname"].split(",")
    head = [int(i.strip()) for i in config["head"].split(",")]

    os.makedirs(resultsroad, exist_ok=True)
    model = CenterNet(head)
    model = t.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()
    model.eval()

    if os.path.exists(weightsroad):
        model.load_state_dict(t.load(weightsroad))
    else:
        print("have no file : {}".format(weightsroad))
        print("please change the term named weightsroad in file config.cfg")
        exit()
    #test_dataset = TestDataset(testroad, (input_w, input_h))
    #testloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=test_dataset.collate_fn)

    color = []
    for i in range(len(classname)):
        color.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    video = cv2.VideoCapture(videoroad)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowrite = cv2.VideoWriter(resultsroad + "Prediction.mp4", fourcc, 20.0, (video.get(3), video.get(4)), True)

    if not video.isOpened():
        print("未能打开该视频")
    else:
        while cv2.waitKey(32) != ord(' '):
            ret, frame = video.read()
            pic = frame.copy()
            if ret:
                cv2.imshow("VideoShow", frame)
                pic = np.transpose(pic, (2, 0, 1))
                pic = test_process(t.from_numpy(pic / 255)).unsqueeze_(0).cuda()
                with t.no_grad():
                     heatmap = model(pic)
                     detection = heatmap_decoding(heatmap[0].cpu().detach().numpy(), heatmap[1].cpu().detach().numpy(),
                                     heatmap[2].cpu().detach().numpy())
                     detection = detection[detection[:, 4] > 0.5]
                     detection = decode_box(frame, detection, (input_w, input_h))
                     detection = nms(detection)

                     frame = draw_box(frame, detection, classname, color)
                     videowrite.write(frame)
    video.close()
    videowrite.close()
if __name__ == "__main__":
    config = config_obtain("config.cfg")
    video_detect(**config)