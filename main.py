import time
import cv2
from pose_estimate import prepare, pose_process
from modules.pre_process import pre_process_openvino

path = 'D:\pycharm\lightweight-human-pose-estimation.pytorch-master\checkpoint_iter_370000.pth'
net = prepare(path)

camera = cv2.VideoCapture(0)
while True:
    _, img = camera.read()
    statr_time = time.time()

    point = pre_process_openvino(img)


    print(1/(time.time() - statr_time))
    cv2.imshow('', img)
    cv2.waitKey(1)