from pypi import MaskDetection
import cv2
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='id of camera or video')
    parser.add_argument('--thred_yolo', type=float, default=0.47, help='thred yolo')
    parser.add_argument('--thred_dis', type=float, default=3, help='thred distance box and hand')
    parser.add_argument('--thred_face', type=float, default=0.5, help='thred face')
    opt = parser.parse_args()
    MaskDetection = MaskDetection()
    MaskDetection.prepare()
    if opt.camera == 0 or opt.camera == 1:
        camera = cv2.VideoCapture(opt.camera)
        while True:
            _, img = camera.read()
            try:
                img = cv2.resize(img, (640, 480))
                pre = MaskDetection.detection(img, opt.thred_yolo, opt.thred_dis, opt.thred_face)
                img = MaskDetection.draw(img)
            except:
                pass
            cv2.imshow('', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(opt.camera)
        try:
            img = cv2.resize(img, (640, 480))
            pre = MaskDetection.detection(img, opt.thred_yolo, opt.thred_dis, opt.thred_face)
            img = MaskDetection.draw(img)
        except:
            pass
        cv2.imshow('', img)
        cv2.waitKey(0)