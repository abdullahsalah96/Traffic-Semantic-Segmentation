from scipy.misc import imsave
from PIL import Image

import numpy as np
import sys
import glob
import os.path as osp
import cv2

input_dir = "/Users/rowanhisham/Documents/QT/MIA2020/Coral-Detection/Dataset/Background"
output_dir = "/Users/rowanhisham/Documents/QT/MIA2020/Coral-Detection/GenerateBackgroundPNG/BackgroundPNG"

for label_file in glob.glob(osp.join(input_dir, '*.jpg')):
        print('Generating dataset from:', label_file)
#        image_array =  np.load(label_file)
        image = cv2.imread(label_file)
        image_array = np.zeros((image.shape[0],image.shape[1]))

        base = osp.splitext(osp.basename(label_file))[0]
        img = Image.fromarray(np.uint8(image_array), 'L')
        dir = osp.join( output_dir, base + '.png')
        img.save(dir)


