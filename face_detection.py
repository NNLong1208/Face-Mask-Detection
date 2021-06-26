import cv2
from openvino.inference_engine import IECore
from modules.pre_process import pre_process_face_detect, pre_process_landmarks
import numpy as np
import time

def face_detection_prepare(path = './models/{}'):
    ie = IECore()
    net = ie.read_network(model=path.format('face_detection.xml'), weights=path.format('face_detection.bin'))
    exec_net_detect = ie.load_network(network=net, device_name="CPU")  # MULTI:GPU,CPU// config={"CPU_THREADS_NUM" : "1"}
    del net
    input_layer_detect = next(iter(exec_net_detect.input_info))
    output_layer_detect = next(iter(exec_net_detect.outputs))

    net = ie.read_network(model=path.format('landmarks.xml'), weights=path.format('landmarks.bin'))
    exec_net_landmarks = ie.load_network(network=net, device_name="CPU")
    del net

    input_layer_landmarks = next(iter(exec_net_landmarks.input_info))
    output_layer_landmarks = next(iter(exec_net_landmarks.outputs))

    para = {'exec_net_detect':exec_net_detect, 'input_layer_detect' : input_layer_detect, 'output_layer_detect': output_layer_detect, 'exec_net_landmarks': exec_net_landmarks, 'input_layer_landmarks': input_layer_landmarks, 'output_layer_landmarks': output_layer_landmarks}
    return para

def face_detection_process(img, para, prob_threshold_face = 0.6):
    frame = pre_process_face_detect(img)
    res_detect = para['exec_net_detect'].infer(inputs={para['input_layer_detect']: frame})
    res_detect = res_detect[
        para['output_layer_detect']]  # [1, 1, N, 7] N là số khuôn mặt format [image_id, label, conf, x_min, y_min, x_max, y_max],

    box_faces = res_detect[0][:, np.where(res_detect[0][0][:, 2] > prob_threshold_face)][0][0]  # Lọc ra các khuôn mặt
    boxes = []
    for face in box_faces:
        box = face[3:7]
        box = [box[i] * img.shape[1] if i % 2 == 0 else box[i] * img.shape[0] for i in range(len(box))]
        box = [int(i) for i in box]
        boxes.append(box)

    landmarks = []
    for box in boxes:
        img_face = img[box[1]:box[3], box[0]:box[2]].copy()
        frame = pre_process_landmarks(img_face.copy())
        res_landmarks = para['exec_net_landmarks'].infer(inputs={para['input_layer_landmarks']: frame})
        res_landmarks = res_landmarks[para['output_layer_landmarks']][0]

        # convert lại landmark
        black_img = np.zeros((img_face.shape[0], img_face.shape[1], 3), np.uint8)

        for i in range(0, len(res_landmarks), 2):
            black_img[int(res_landmarks[i + 1][0][0] * img_face.shape[0]), int(
                res_landmarks[i][0][0] * img_face.shape[1])] = [0, 255, 0]

        landmark = np.argwhere(black_img[:, :, 1] != 0)

        for i in landmark:
            tempt = i[0] + box[1]  # chuyển lại về dạng (x,y)
            i[0] = i[1] + box[0]
            i[1] = tempt
        landmarks.append(landmark.tolist())

    boxes_conver = [[[x1, y1], [x2, y2]] for x1, y1, x2, y2 in boxes]
    return boxes_conver, landmarks

if __name__ == '__main__':
    para = face_detection_prepare()
    cam = cv2.VideoCapture(0)
    while True:
        _, img = cam.read()
        img = cv2.resize(img, (640, 480))
        time_start = time.time()
        box, landmarks = face_detection_process(img, para)
        print(1/(time.time() - time_start))
        cv2.imshow('', img)
        cv2.waitKey(1)