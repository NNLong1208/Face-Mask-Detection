from pose_estimate import prepare, pose_process
from mask_recognition import mask_prepare, mask_process
import time
from mask_detection import *
import cv2
from utils.torch_utils import select_device

path_pose = '.\models\pose_estimate.pth'
path_open = '.\models'
net_pose = prepare(path_pose)
net_open, input_layer, output_layer = mask_prepare(path_open)
weights = r'.\models\yolov5.pt'
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA - TRUE
model = yolo_prepare(weights, device)
if half:
    model.half()  # to FP16
camera = cv2.VideoCapture(0)
while True:
    _, img = camera.read()
    statr_time = time.time()

    point = pose_process(img, net_pose)
    a = mask_process([img], net_open, input_layer, output_layer)
    b = yolo_process(img, model, device, half)

    print(1/(time.time() - statr_time))
    cv2.imshow('', img)
    cv2.waitKey(1)