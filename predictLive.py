from keras.models import load_model
from helper import Segmentation
import numpy as np
from keras.preprocessing import image
import cv2
from threading import Thread, Timer

im = Segmentation()
loaded_model = load_model('Coral_Model.h5')
frame = np.zeros((128,128,3), dtype = np.uint8)

def predict_class(model, image):
    """
    A funtion that takes the model and the path of the image to be predicted and returns the prediction
    """
    # print(image.shape)
    # image.reshape(1, 128, 128, 3)
    # image = np.expand_dims(image, axis=0)/255.0
    # image = np.expand_dims(image, axis=3)
    # print(image)
    # print(image.shape)
    # prediction = model.predict(image)
    image = np.expand_dims(image, axis=0)/255.0
    prediction = model.predict(image)
    return np.reshape(prediction,(128,128,2))


class Thread_Prediction():
    def __init__ (self):
        self.model = load_model('Coral_Model.h5')
        self.frame = np.zeros((128,128,3), dtype = np.uint8)
        self.out = np.zeros((128,128,1), dtype = np.uint8)
        self.segmentation_map = np.zeros((128,128,3), np.uint8)
        self.thread = Thread(target = self.timer_fn)
        self.timer = Timer(0.5, self.predict_class)

    def start_thread(self):
        self.thread.setDaemon(True)
        self.thread.start()
        self.thread.join()

    def timer_fn(self):
        self.timer.start()

    def predict_class(self):
        """
        A funtion that takes the model and the path of the image to be predicted and returns the prediction
        """
        # print(image.shape)
        # image.reshape(1, 128, 128, 3)
        # image = np.expand_dims(image, axis=0)/255.0
        # image = np.expand_dims(image, axis=3)
        # print(image)
        # print(image.shape)
        # prediction = model.predict(image)
        image = self.frame
        image = np.expand_dims(image, axis=0)/255.0
        prediction = self.model.predict(image)
        prediction = np.reshape(prediction,(128,128,2))
        for i in range(128):
            for j in range(128):
                self.out[i][j] = np.argmax(prediction[i][j])
                if(self.out[i][j] == 0):
                    self.segmentation_map[i,j] = (0,255,0)
                elif(self.out[i][j] == 1):
                    self.segmentation_map[i,j] = (0,0,255)
        print("modified seg")


# cap = cv2.VideoCapture("/Users/rowanhisham/Documents/QT/ROBOT-PID-Control/2020GUI/MappingVideos/IMG_4459.MOV")
cap = cv2.VideoCapture(0)
th = Thread_Prediction()
# th.start_thread()

while 1:
    ret, frame2 = cap.read()
    frame = cv2.resize(frame2, (128, 128))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prediction = predict_class(loaded_model, frame)
    out = np.zeros((128,128,1), dtype = np.uint8)
    segmentation_map = np.zeros((128,128,3), np.uint8)
    # #getting the maximum pixel value of the one hot encoded array
    for i in range(128):
        for j in range(128):
            out[i][j] = np.argmax(prediction[i][j])
            # print(prediction[i][j])
            if(out[i][j] == 0):
                segmentation_map[i,j] = (0,255,0)
            elif(out[i][j] == 1):
                segmentation_map[i,j] = (0,0,255)


    frame = cv2.resize(frame, (480, 360))
    out = cv2.resize(out, (480,360))
    # segmentation_map = th.segmentation_map
    segmentation_map = cv2.resize(segmentation_map, (480,360))
    # vis_image = cv2.addWeighted(frame,1.0,segmentation_map,0.5,0)
    cv2.imshow('src', frame)
    cv2.imshow('map', segmentation_map)
    # cv2.imshow('visualization', vis_image)

    if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite("s.jpg", frame2)
        cv2.imwrite("s.png", frame2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
