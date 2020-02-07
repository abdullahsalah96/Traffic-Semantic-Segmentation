import cv2
import os
import math
INPUT_VIDEO = "/Users/abdallaelshikh/Desktop/MIA/Coral-Detection/Dataset/VID_20200130_171307.mp4"
OUTPUT_PATH = "/Users/abdallaelshikh/Desktop/MIA/Coral-Detection/Dataset/Images"
os.makedirs(OUTPUT_PATH, exist_ok=True)
SAVE_FPS_RATE = 0.75

vidcap = cv2.VideoCapture(INPUT_VIDEO)
success,image = vidcap.read()
image_num = 3059
fps = int(vidcap.get(cv2.CAP_PROP_FPS)*SAVE_FPS_RATE)
while success:
    if image_num%(fps) == 0 :
        cv2.imwrite(OUTPUT_PATH + "/image%04i.jpg" %image_num, image)
        print("Saving image: " + OUTPUT_PATH + "/image%04i.jpg" %image_num)
    success,image = vidcap.read()
    image_num += 1

vidcap.release()
