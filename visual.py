import numpy as np
import cv2
def panel(name, color):
    pan = np.ones((750, 1050, 3))
    cv2.line(pan, (30, 710), (1030,710), (0, 0, 0), 2)
    cv2.line(pan, (30, 10), (30,710), (0, 0, 0), 2)
    for i in range(30, 1030, 100):
       cv2.line(pan, (i, 710), (i, 700), (0, 0, 0), 2)
       cv2.putText(pan, str(i-30), (i-10, 730), cv2.FONT_HERSHEY_SCRIPT_COMPLEX  , 0.6, color, 2)
    for i in range(730, 30, -100):
       cv2.line(pan, (30, i), (50, i), (0, 0, 0), 2)
       cv2.putText(pan, str(((730 - i)) // 100), (5, i), cv2.FONT_HERSHEY_SCRIPT_COMPLEX  , 0.6, color, 2)
    cv2.putText(pan, name, (500, 30), cv2.FONT_HERSHEY_TRIPLEX , 1, color, 2)
    return pan
    #cv2.imshow("win", pan)
    #cv2.waitKey(0)
panel("title", (255, 0, 0))