from keras.models import load_model
from helper import Segmentation
import numpy as np
from keras.preprocessing import image
import cv2
from threading import Thread, Timer
from config import *
import time

im = Segmentation()
loaded_model = load_model('UNET20.h5')
frame = np.zeros((WIDTH,HEIGHT,COLOR_DEPTH), dtype = np.uint8)
cap = cv2.VideoCapture(0)

def predict_class(model, image):
    """
    A funtion that takes the model and the path of the image to be predicted and returns the prediction
    """
    image = np.expand_dims(image, axis=0)/255.0
    prediction = model.predict(image)
    return np.reshape(prediction,(WIDTH,HEIGHT,NUM_OF_CLASSES))

while 1:
    ret, frame2 = cap.read()
    start_time = time.time() # start time of the loop
    frame = cv2.resize(frame2, (WIDTH, HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prediction = predict_class(loaded_model, frame)
    out = np.zeros((WIDTH,HEIGHT,1), dtype = np.uint8)
    segmentation_map = np.zeros((WIDTH,HEIGHT,COLOR_DEPTH), np.uint8)
    for i in range(WIDTH):
        for j in range(HEIGHT):
            out[i][j] = np.argmax(prediction[i][j])
            class_num = out[i][j]
            class_num = class_num[0]
            segmentation_map[i,j] = Cityscapes_Colors[class_num]

    frame = cv2.resize(frame, (480, 360))
    out = cv2.resize(out, (480,360))
    # segmentation_map = th.segmentation_map
    segmentation_map = cv2.resize(segmentation_map, (480,360))
    # vis_image = cv2.addWeighted(frame,1.0,segmentation_map,0.5,0)
    # cv2.imshow('src', frame)
    cv2.imshow('map', segmentation_map)
    # cv2.imshow('out', out*255)
    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

    if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite("s.jpg", frame2)
        cv2.imwrite("s.png", frame2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
