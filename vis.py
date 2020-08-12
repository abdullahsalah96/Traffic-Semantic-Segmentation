import cv2
import numpy as np

path = '/Users/abdallaelshikh/Desktop/College/Graduation Project/Semantic Segmentation Datasets/CityScapes/Dataset/Train/Annotations/image0000.png'

img = cv2.imread(path, 0)
print(np.max(img))