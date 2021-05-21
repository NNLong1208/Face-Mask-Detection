from pose_estimate_ver2 import pose_prepare, pose_process
from mask_recognition import mask_prepare, mask_process
from mask_detection import *
import cv2
from utils.torch_utils import select_device
from modules.modules import *
from modules import download

class MaskDetection_1p:
    def __init__(self,):
        self.__net_open = None
        self.__net_yolo = None
        self.__net_pose = None
        self.__input_layer = None
        self.__output_layer = None
        self.__device = None
        self.output = None

    def load_model(self, path_yolo, path_open):
        print("Loading Model")
        self.__net_open, self.__input_layer, self.__output_layer = mask_prepare(path_open, device='GPU')
        self.__device = select_device('0')
        self.__net_yolo = yolo_prepare(path_yolo, self.__device)
        self.__net_yolo.half()
        self.__net_pose = pose_prepare()
        print('Load Model Success')

    def prepare(self, path_yolo=r'./models/yolov5.pt', path_open=r'./models'):
        try:
            self.load_model(path_yolo, path_open)
        except:
            download.dowload()
            self.load_model(path_yolo, path_open)

    def detection(self, img, thred = 2.1):
        assert (img.shape == (480, 640, 3)), "Use cv2.resize(img, (640, 480) befor predict"

        faces, boxes_face, hand = pose_process(img, self.__net_pose)
        yolo_pre = yolo_process(img, self.__net_yolo, self.__device, True)[0]
        yolo_pre = sort_mask(yolo_pre, boxes_face)
        res = mask_process(faces, self.__net_open, self.__input_layer, self.__output_layer)
        check_dis = check_distance(boxes_face, yolo_pre, hand, thred)
        results = get_results(res, check_dis, yolo_pre)
        mask_detection_results = []
        for box_face, yolo, result in zip(boxes_face, yolo_pre, results):
            if result == 1:
                mask_detection_results.append({'box': [[int(yolo[0]), int(yolo[1])], [int(yolo[2]), int(yolo[3])]],
                                               'acc': round(float(yolo[4]), 2), 'label': 'Mask'})
            else:
                mask_detection_results.append({'box': box_face, 'acc': None, 'label': 'No Mask'})
        self.output = mask_detection_results
        return mask_detection_results

    def draw(self, img):
        if self.output != None:
            for ele in self.output:
                box = ele['box']
                if ele['label'] == 'Mask':
                    cv2.rectangle(img, box[0], box[1], (0, 255, 0), 2)
                    cv2.putText(img, str(ele['acc']), (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1)
                    cv2.putText(img, str(ele['label']), (box[0][0], box[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1)
                else:
                    cv2.rectangle(img, box[0], box[1], (0, 0, 255), 2)
                    cv2.putText(img, str(ele['acc']), (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                    cv2.putText(img, str(ele['label']), (box[0][0], box[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        return img

mask_detection = MaskDetection_1p()
mask_detection.prepare()
camera = cv2.VideoCapture(0)
while True:
    _, img = camera.read()
    pre = mask_detection.detection(img)
    img = mask_detection.draw(img)
    cv2.imshow('', img)
    cv2.waitKey(1)