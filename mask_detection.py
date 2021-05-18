import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from modules.pre_process import pre_process_yolo
import cv2

def yolo_prepare(weights, device):
    imgsz = 640
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    cudnn.benchmark = True  # set True to speed up constant image size inference
    return model

def yolo_process(img, model, device, half):
    img = cv2.resize(img, (640, 480))
    img_ = pre_process_yolo(img, device, half)
    pred = model(img_, augment=False)[0]
    pred = non_max_suppression(pred, 0.75, 0.45, classes=None, agnostic=True)
    for box in pred[0]:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(img, str(round(float(box[4]), 2)), (int(box[0] + 10), int(box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img