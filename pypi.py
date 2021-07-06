from pose_estimate import prepare_pose, pose_process
from mask_recognition import mask_prepare, mask_process
from mask_detection import *
from utils.torch_utils import select_device
from modules.modules import *
from face_detection import *
from modules import download
class MaskDetection:
    def __init__(self,):
        self.__net_open = None
        self.__net_yolo = None
        self.__net_pose = None
        self.__input_layer = None
        self.__output_layer = None
        self.__device = None
        self.__para_net_face = None
        self.output = None

    def load_model(self, path_yolo, path_pose, path_open, path_face):
        print("Loading Model")
        self.__net_open, self.__input_layer, self.__output_layer = mask_prepare(path_open, device='GPU')
        self.__device = select_device('0')
        self.__net_yolo = yolo_prepare(path_yolo, self.__device)
        self.__net_yolo.half()
        self.__net_pose = prepare_pose(path_pose)
        self.__para_net_face = face_detection_prepare(path_face)
        self.detection(np.random.randint(255, size=(480, 640, 3),dtype=np.uint8),test = 1) #test model
        print('Load Model Success')

    def prepare(self, path_yolo=r'./models/yolov5.pt', path_pose='./models/pose_estimate.pth', path_open=r'./models', path_face = './models/{}'):
        try:
            self.load_model(path_yolo, path_pose, path_open, path_face)
        except:
            download.dowload()
            self.load_model(path_yolo, path_pose, path_open, path_face)

    def detection(self, img, thred_yolo = 0.47 ,thred_dis = 3,thred_face= 0.5, test = 0):
        assert(img.shape == (480, 640, 3)),"Use cv2.resize(img, (640, 480) befor predict"

        mask_detection_results = []
        boxes_face, landmarks = face_detection_process(img, self.__para_net_face, prob_threshold_face=thred_face)
        if len(boxes_face) == 0 and test == 1:
            return mask_detection_results
        res = mask_process(img, landmarks, self.__net_open, self.__input_layer, self.__output_layer)
        if sum(res) == 0 and test == 1:
            return mask_detection_results
        point = pose_process(img, self.__net_pose)
        faces, box, hand = get_face_box(img, point)
        hand = combine_box(box, boxes_face,hand)
        yolo_pre = yolo_process(img, self.__net_yolo, self.__device, True, thred=thred_yolo)[0]
        print('Face', boxes_face)
        print('Open', res)
        print('Mask', yolo_pre)
        yolo_pre = sort_mask(yolo_pre, boxes_face)
        check_dis = check_distance(boxes_face, yolo_pre, hand, thred_dis)

        print('Sort', yolo_pre)
        print('Dis ', check_dis)
        print('-------------------------------------------------------')
        results = get_results(res, check_dis, yolo_pre)
        for box_face, yolo, result in zip(boxes_face, yolo_pre, results):
            if result == 1:
                mask_detection_results.append({'box': [[int(yolo[0]), int(yolo[1])], [int(yolo[2]), int(yolo[3])]], 'acc': round(float(yolo[4]), 2), 'label': 'Mask'})
            else:
                mask_detection_results.append({'box': box_face, 'acc': None, 'label': 'No Mask'})
        self.output = mask_detection_results
        return mask_detection_results

    def draw(self, img):
        if self.output != None:
            for ele in self.output:
                box = ele['box']
                #if 2.4*(box[1][0] - box[0][0]) > (box[1][1] - box[0][1]):
                if ele['label'] == 'Mask':
                    cv2.rectangle(img, box[0], box[1], (0, 255, 0), 2)
                    cv2.putText(img, str(ele['acc']), (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1)
                    cv2.putText(img, str(ele['label']), (box[0][0], box[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1)
                else:
                    cv2.rectangle(img, box[0], box[1], (0, 0, 255), 2)
                    cv2.putText(img, str(ele['acc']), (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                    cv2.putText(img, str(ele['label']), (box[0][0], box[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        return img

