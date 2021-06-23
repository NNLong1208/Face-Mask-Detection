import torch
import cv2
import numpy as np
def pre_process_yolo(img, device, half):
    img_ = img[..., ::-1].copy()
    img_ = torch.from_numpy(img_).to(device)
    img_ = img_.half().to('cuda') if half else img_.float()
    img_ /= 255.0
    img_ = img_.permute(2, 0, 1)
    img_ = img_.unsqueeze(0)
    return img_

def pre_process_openvino(x):
    x = cv2.resize(x, (120, 120))
    x = torch.from_numpy(x).cuda()
    x = x.to(dtype=torch.float)
    x /= 255.0
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.cpu().numpy()
    return x

def pre_process_face_detect(img):
    img = cv2.resize(img, (300, 300))
    img = np.transpose(img, (2, 0, 1))
    img = img.reshape(1, 3, 300, 300)
    return img

def pre_process_landmarks(img):
    img = cv2.resize(img,(48,48))
    img = np.transpose(img, (2, 0, 1))
    img = img.reshape(1, 3, 48, 48)
    return img
