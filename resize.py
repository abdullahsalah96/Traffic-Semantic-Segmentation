import cv2
import glob
import os

INPUT_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/Semantic Segmentation Datasets/CityScapes/Validation/Images"
OUTPUT_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/Semantic Segmentation Datasets/CityScapes/ResizedDataset/Validation/Images"
RESIZE_FACTOR = 3
image_num = 0000

os.makedirs(OUTPUT_PATH, exist_ok=True)

for img in sorted(glob.glob(INPUT_PATH + "/*.png")):
    print("RESIZING IMAGE " + str(image_num))
    image = cv2.imread(img)
    resized = cv2.resize(image,(int(image.shape[1]/RESIZE_FACTOR), int(image.shape[0]/RESIZE_FACTOR)))
    cv2.imwrite(OUTPUT_PATH + "/image%04i.png" %image_num, resized)
    if(image_num == 6509):
        break
    image_num+=1

print("SAVED RESIZED IMAGES")
