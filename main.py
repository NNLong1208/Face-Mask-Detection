from pypi import MaskDetection
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='id of camera or video')
    opt = parser.parse_args()
    MaskDetection = MaskDetection()
    MaskDetection.prepare()
    camera = cv2.VideoCapture(opt.camera)
    while True:
        _, img = camera.read()
        img = cv2.resize(img, (640, 480))
        pre = MaskDetection.detection(img, thred=3)
        img = MaskDetection.draw(img)
        cv2.imshow('', img)
        cv2.waitKey(1)
    camera.release()
    cv2.destroyAllWindows()