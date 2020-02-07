import cv2
import glob
import os

INPUT_PATH = "/Users/abdallaelshikh/Desktop/MIA/Coral-Detection/coral(warsha)/photos/Images"
OUTPUT_PATH = "/Users/abdallaelshikh/Desktop/MIA/Coral-Detection/coral(warsha)/photos/Resized"
RESIZE_FACTOR = 5
os.makedirs(OUTPUT_PATH, exist_ok=True)
image_num = 0

for img in glob.glob(INPUT_PATH + "/*.jpg"):
    print("RESIZING IMAGE " + str(image_num))
    image = cv2.imread(img)
    resized = cv2.resize(image,(int(image.shape[1]/RESIZE_FACTOR), int(image.shape[0]/RESIZE_FACTOR)))
    cv2.imwrite(OUTPUT_PATH + "/image%04i.jpg" %image_num, resized)
    image_num+=1

print("SAVED RESIZED IMAGES")

