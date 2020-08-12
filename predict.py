from keras.models import load_model
from helper import Segmentation
import numpy as np
import cv2
from config import *

im = Segmentation()

def predict_class(model, image):
    """
    A funtion that takes the model and the path of the image to be predicted and returns the prediction
    """
    image = np.expand_dims(image, axis=0)/255.0
    prediction = model.predict(image)
    return np.reshape(prediction,(WIDTH,HEIGHT,NUM_OF_CLASSES))
    
PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/Semantic Segmentation Datasets/CityScapes/Dataset/Test/Images/image0000.png"

loaded_model = load_model("UNET20.h5")

img = cv2.imread(PATH)
frame = cv2.resize(img, (WIDTH, HEIGHT))
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

out = np.uint8(out.reshape((WIDTH, HEIGHT, 1)))


src = img
src = cv2.resize(src, (480,360))
out = cv2.resize(out, (480,360))
segmentation_map = cv2.resize(segmentation_map, (480,360))
vis_image = cv2.addWeighted(src,1.0,segmentation_map,0.5,0)
cv2.imshow('src', src)
cv2.imshow('map', segmentation_map)
cv2.imshow('vis', vis_image)
cv2.waitKey(0)