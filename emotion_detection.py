import cv2
from fer import FER

fer = FER()

img = cv2.imread("happy_man.jpg")
result = fer.predict_image(img, show_top = True)