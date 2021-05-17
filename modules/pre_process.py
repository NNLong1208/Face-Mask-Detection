import torch
import cv2

def pre_process_yolo(img):
    pass

def pro_process_openvino(x):
    x = cv2.resize(x, (120, 120))
    x = torch.from_numpy(x).cuda()
    x = x.to(dtype=torch.float)
    x /= 255.0
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.cpu().numpy()
    return x
