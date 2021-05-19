from pose_estimate import prepare, pose_process
from mask_recognition import mask_prepare, mask_process
import time
from mask_detection import *
import cv2
from utils.torch_utils import select_device
from modules.modules import *
path_pose = '.\models\pose_estimate.pth'
path_open = '.\models'
net_pose = prepare(path_pose)
net_open, input_layer, output_layer = mask_prepare(path_open, device='CPU')
weights = r'.\models\yolov5.pt'
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA - TRUE
model = yolo_prepare(weights, device)
if half:
    model.half()  # to FP16
camera = cv2.VideoCapture(0)
while True:
    _, img = camera.read()
    img = cv2.resize(img, (640, 480))
    statr_time = time.time()
    point = pose_process(img, net_pose)
    for i in point:
        for j in i:
            cv2.circle(img, j, 1, (0,255,0), 2)
    faces, boxes_face, hand = get_face_box(img, point)

    b = yolo_process(img, model, device, half)[0]
    mask = sort_mask(b, boxes_face)
    res = mask_process(faces, net_open, input_layer, output_layer)
    check_dis = check_distance(boxes_face, mask, hand)
    results = get_results(res, check_dis, mask)

    for box_face, box_mask, res in zip(boxes_face, mask, results):
        if res == 1:
            cv2.rectangle(img, (int(box_mask[0]), int(box_mask[1])), (int(box_mask[2]), int(box_mask[3])), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, box_face[0], box_face[1], (0, 0, 255), 2)
    #print(1/(time.time() - statr_time))
    #cv2.imshow('1', faces[0])
    cv2.imshow('', img)
    cv2.waitKey(1)