from pypi import MaskDetection
import cv2
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='id of camera or video')
    parser.add_argument('--thred_yolo', type=int, default=0.47, help='thred yolo')
    parser.add_argument('--thred_dis', type=int, default=3, help='thred distance box and hand')
    opt = parser.parse_args()
    MaskDetection = MaskDetection()
    MaskDetection.prepare()
    camera = cv2.VideoCapture(opt.camera)
    while True:
        _, img = camera.read()
        img = cv2.resize(img, (640, 480))
        try:
            pre = MaskDetection.detection(img, opt.thred_yolo, opt.thred_dis)
            img = MaskDetection.draw(img)
        except:
            pass
        cv2.imshow('', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()