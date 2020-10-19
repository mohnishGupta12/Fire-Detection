# import tensorflow.keras as keras
import sys
import cv2
from PIL import Image, ImageDraw , ImageFont
from tensorflow.keras.models import model_from_json
import numpy as np

categories = ['Fire' , 'no Fire']
#loading the json_file
flag = 0

def load_model():
    print('[INFO] loading the model')
    load_json = open('model/fire_detection.json' , 'r')
    load_fire_detection_model = load_json.read()
    fire_detection_model = model_from_json(load_fire_detection_model)
    fire_detection_model.load_weights('model/fire_detection.h5')
    print('[INFO] model loaded')
    return fire_detection_model

def webcam():
    #load the face fire detection model
    fire_detection_model = load_model()
    x,y,w,h = 10,10,450,120
    cap = cv2.VideoCapture('videos/videos.mp4')
    cv2.namedWindow('fire detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('fire detection', 800, 800)
    while cap:
        ret , frame = cap.read()
        resize = cv2.resize(frame , (224 , 224))
        reshape = resize.reshape(1,224,224,3)
        pred = fire_detection_model(reshape)
        if categories[np.argmax(pred)] == 'Fire':
            flag = flag +1
            if flag >3:
                frame = cv2.rectangle(frame, (x, x), (x + w, y + h), (0,0,0), -1)
                frame = cv2.putText(frame, "ALERT-FIRE", (x + int(w/10),y + int(h/2)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        else:
            flag = 0
        cv2.imshow('fire detection' , frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # cap.release()
        # cv2.destroyAllWindows()

webcam()
