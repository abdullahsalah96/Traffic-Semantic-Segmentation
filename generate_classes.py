import json
import cv2
import glob
import os
import numpy as np

JSON_PATH = "/Volumes/EXTERNAL/mapillary-vistas-dataset_public_v1.1/config.json"
NUM_OF_CLASSES = 34

with open(JSON_PATH) as f:
  data = json.load(f)

labels = data['labels']

classes = dict.fromkeys(range(len(labels)), []) 
classes_names = dict.fromkeys(range(len(labels)), []) 

for class_num, label in enumerate(labels):
    class_name = label['readable']
    class_color = list(label['color'])
    classes[class_num] = class_color
    classes_names[class_num] = class_name

print(classes)
print(classes_names)

INPUT_PATH = "/Volumes/EXTERNAL/mapillary-vistas-dataset_public_v1.1/training/Resized/Labels"
OUTPUT_PATH = "/Volumes/EXTERNAL/mapillary-vistas-dataset_public_v1.1/training/Resized/Annotations"
image_num = 0000

os.makedirs(OUTPUT_PATH, exist_ok=True)

for img in sorted(glob.glob(INPUT_PATH + "/*.png")):
    print("Generating annotation: " + str(image_num))
    image = cv2.imread(img)
    out = np.zeros((image.shape[0],image.shape[1],3), dtype = np.uint8)
    for class_num, color in classes.items:
        out[:,:] = np.where(image[:,:] == color, class_num, out[:,:]) 
    # out = np.uint8(out.reshape((image.shape[1], image.shape[0], 1)))
    print(str(image[350,190]))
    print(out[350,190])
    
    if(image_num==0):
        break
    # cv2.imwrite(OUTPUT_PATH + "/image%04i.png" %image_num, resized)
    image_num+=1

print("DONE CONVERTING IMAGES")
