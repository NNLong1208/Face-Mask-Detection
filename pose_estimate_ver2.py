import cv2
from modules.modules import *
import mediapipe as mp

def pose_prepare(de_conf = 0.5, track_conf = 0.5, model_com = 1):
    mp_pose = mp.solutions.pose
    model = mp_pose.Pose(min_detection_confidence=de_conf, min_tracking_confidence=track_conf, model_complexity=model_com)
    return model

def pose_process(img, model):
    id_hand = [15, 16, 17, 18, 19, 20, 21, 22]
    h, w, c = img.shape
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    point = []
    box_face = [-1, -1, -1, -1]
    face = []
    try:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if lm.visibility > 0.7:
                if id in id_hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    point.append([cx, cy])
                if id == 6:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    box_face[1] = cy
                if id == 8:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    box_face[0] = cx
                if id == 7:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    box_face[2] = cx
                if id == 9:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    box_face[3] = cy
        w = box_face[2] - box_face[0]
        h = box_face[3] - box_face[1]

        box_face[0] -= w//7
        box_face[2] += w//7
        box_face[1] -= h//2
        box_face[3] += int(h/1.2)
        img_crop = img[box_face[0]:box_face[2], box_face[1]:box_face[3]].copy()
        face.append(img_crop)
    except:
        pass
    if len(point) == 0:
        point.append([5000,5000])
    point = np.array(point).reshape(-1,2)
    box_face = [box_face]
    return box_face, point, face
