import torch
import cv2
from utils.torch_utils import select_device

def pre_process_yolo(img):
    img_ = img[..., ::-1].copy()
    img_ = torch.from_numpy(img_).to(select_device('0'))
    img_ = img_.half()
    img_ /= 255.0
    img_ = img_.permute(2, 0, 1)
    img_ = img_.unsqueeze(0)
    return img_

def pro_process_openvino(x):
    x = cv2.resize(x, (120, 120))
    x = torch.from_numpy(x).cuda()
    x = x.to(dtype=torch.float)
    x /= 255.0
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.cpu().numpy()
    return x
