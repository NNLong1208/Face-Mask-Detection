import numpy as np

def get_face_box(img, points):
    boxes = []
    faces = []
    hand = []
    for point in points:
        if point[1] != -1 and point[14] != -1 and point[15] != -1:
            box = [[point[14][0], point[14][1]], [point[15][0], point[1][1]]]

            w_box = box[1][0] - box[0][0]
            h_box = box[1][1] - box[0][1]

            box[0][1] -= int(h_box // 5)
            box[1][1] -= int(h_box // 2.5)
            box[0][0] -= w_box // 2
            box[1][0] += w_box // 2
            if box[1][1] - box[0][1] > 0 and box[1][0] - box[0][0] > 0:
                h = np.array([point[4], point[7]])
                hand.append(h)
                boxes.append(box)
                img_crop = img[box[0][1]:box[1][1], box[0][0]:box[1][0]].copy()
                faces.append(img_crop)
    return faces, boxes, np.array(hand)

def sort_mask(pre, boxes_face):
    mask = [[-1,-1,-1,-1,-1,-1] for x in range(len(boxes_face))]
    for ele in pre:
        center = np.array([int((int(ele[0]) + int(ele[2])) / 2), int((int(ele[1]) + int(ele[3])) / 2)])
        for id, box_face in enumerate(boxes_face):
            if box_face[0][1] < center[1] < box_face[1][1] and box_face[0][0] < center[0] < box_face[1][0]:
                mask[id] = ele
    return mask

def get_distance(a, b):
    a = np.array([[5000,5000] if x[0] == -1 else x for x in a])
    return np.sum((a - b) ** 2, axis=1) ** (1 / 2)

def check_distance(boxes_face, masks, hands, thred = 3):
    res_list = []
    for boxe_face, mask, hand in zip(boxes_face, masks, hands):
        center = np.array([int((int(mask[0]) + int(mask[2])) / 2), int((int(mask[1]) + int(mask[3])) / 2)])
        dis = ((center[0] - int(boxe_face[0][0])) ** 2 + (center[1] - int(boxe_face[1][1])) ** 2) ** (1 / 2)
        distance = get_distance(hand, center)
        if np.min(distance) / dis > thred:
            res_list.append(1)
        else:
            res_list.append(0)
    return res_list

def get_results(open, check_distance, mask):
    mask_res = [1 if x[0] != 1 else 0 for x in mask]
    res = [1 if x*2+y+z == 4 else 0 for x, y, z in zip(open, check_distance, mask_res)]
    return res

def get_iou(bb1, bb2):

    x_left = max(bb1[0][0], bb2[0][0])
    y_top = max(bb1[0][1], bb2[0][0])
    x_right = min(bb1[1][0], bb2[1][0])
    y_bottom = min(bb1[1][1], bb2[1][1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[1][0] - bb1[0][0]) * (bb1[1][1] - bb1[0][1])
    bb2_area = (bb2[1][0] - bb2[0][0]) * (bb2[1][1] - bb2[0][1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def combine_box(box_pose, box_detect):
    box = []
    for ele_pose in box_pose:
        IoU = []
        for ele_detect in box_detect:
            IoU.append(get_iou(ele_pose, ele_detect))
        if len(IoU) != 0:
            box.append(box_detect[IoU.index(max(IoU))])
    return box