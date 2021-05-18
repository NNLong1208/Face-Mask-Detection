import time
from mask_detection import *
import cv2
from utils.torch_utils import select_device

weights = r'./models/yolov5.pt'
device = select_device('cpu')
half = device.type != 'cpu'  # half precision only supported on CUDA - TRUE
model = yolo_prepare(weights, device)
if half:
    model.half()  # to FP16

camera = cv2.VideoCapture(0)
while True:
    _, img = camera.read()
    start_time = time.time()
    img = yolo_process(img, model, device, half)
    fps = 1/(time.time() - start_time)
    cv2.putText(img, str(round(float(fps), 2)), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('', img)
    cv2.waitKey(1)

