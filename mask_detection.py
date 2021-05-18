import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression
from modules.pre_process import pre_process_yolo
import cv2

def yolo_prepare(weights, device):
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    cudnn.benchmark = True  # set True to speed up constant image size inference
    return model

def yolo_process(img, model, device, half, thred = 0.4):
    img = cv2.resize(img, (640, 480))
    img_ = pre_process_yolo(img, device, half)
    pred = model(img_, augment=False)[0]
    pred = non_max_suppression(pred, thred, 0.45, classes=None, agnostic=True)
    return pred