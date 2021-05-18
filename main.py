import time
import cv2
from pose_estimate import pose_prepare, pose_process
from mask_recognition import mask_prepare, mask_process

path_pose = r'D:\pycharm\lightweight-human-pose-estimation.pytorch-master\checkpoint_iter_370000.pth'
path_mask = r'D:\pycharm\NCKH-NEU\models\FP16'
net = pose_prepare(path_pose)
net_mask, input_layer, output_layer = mask_prepare(path_mask)
camera = cv2.VideoCapture(0)
while True:
    _, img = camera.read()
    statr_time = time.time()
    a = mask_process(img, net_mask, input_layer, output_layer)
    print(a)
    poses, point = pose_process(img, net)

    print(1/(time.time() - statr_time))
    cv2.imshow('', img)
    cv2.waitKey(1)