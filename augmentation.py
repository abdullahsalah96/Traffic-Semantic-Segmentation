import cv2
from helper import Segmentation
import random
from sklearn.datasets import load_files
import numpy as np
# import skimage
import glob
import os
import os.path as osp

class Augmentation():
    def __init__(self, input_path, output_images_path, input_annotations_path, output_annotations_path, num_of_augemntations, color_shift, contrast_shift, brightness_shift, blur):
        self.input_path = input_path
        self.output_images_path = output_images_path
        self.input_annotations_path = input_annotations_path
        self.output_annotations_path = output_annotations_path
        self.num_of_augemntations = num_of_augemntations
        self.color_shift = color_shift
        self.contrast_shift = contrast_shift
        self.brightness_shift = brightness_shift
        self.blur = blur

    def load_images(self, files_path, ext):
        """
        A funtion that takes the path of the images and returns a numpy array containing the images paths and a numpy array of one hot encoded labels
        """
        # data = load_files(files_path, shuffle = False) #load files
        # images = np.array(data['filenames']) #load images
        data = sorted(glob.glob(osp.join(files_path, ext)))
        images = np.array(data)  # load images
        return images

    def augment(self, color_range, contrast_range, brightness_range):
        print("Loading Images....")
        self.images = self.load_images(self.input_path,"*.jpg")
        print(self.images.shape)
        print("Loading Annotations....")
        self.annotations = self.load_images(self.input_annotations_path,"*.png")
        print(self.annotations.shape)
        out_img_num = 50000
        for number,path in enumerate(self.images): 
            annotation_img = cv2.imread(self.annotations[number])
            count = 0
            for num in range(self.num_of_augemntations):
                image = cv2.imread(path)
                # print(path)
                if(self.contrast_shift):
                    val = random.randint(-contrast_range, contrast_range)
                    print("contrast: " +str(val))
                    image = self.apply_brightness_contrast(image, brightness = 1, contrast = val)

                if(self.brightness_shift):
                    val = random.randint(-brightness_range, brightness_range)
                    print("brightness: " +str(val))
                    image = self.apply_brightness_contrast(image,brightness = val, contrast = 1)

                if(self.color_shift):
                    b, g, r = cv2.split(image)
                    if(count == 0):
                        b_val = random.randint(-color_range, color_range)
                        print("b: " + str(b_val))
                        cv2.add(b,b_val, b)
                        image = cv2.merge((b,g,r))
                    elif(count == 1):
                        g_val = random.randint(-color_range, color_range)
                        print("g: " + str(g_val))
                        cv2.add(g,g_val, g)
                        image = cv2.merge((b,g,r))
                    elif(count == 2):
                        r_val = random.randint(-color_range, color_range)
                        print("r: " + str(r_val))
                        cv2.add(r,r_val, r)
                    else:
                        count = 0
                    image = cv2.merge((b,g,r))

                if(self.blur):
                    image = cv2.GaussianBlur(image,(5,5),1)
                out_img_num+=num
                self.save_image(self.output_images_path, out_img_num, image,".jpg")
                self.save_image(self.output_annotations_path, out_img_num, annotation_img,".png")
                count +=1
                print("===============================================")

        print("Finished Augmentation")

    def apply_brightness_contrast(self, input_img, brightness = 255, contrast = 127):
            if brightness != 0:
                if brightness > 0:
                    shadow = brightness
                    highlight = 255
                else:
                    shadow = 0
                    highlight = 255 + brightness
                alpha_b = (highlight - shadow)/255
                gamma_b = shadow

                buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
            else:
                buf = input_img.copy()

            if contrast != 0:
                temp = (127*(131-contrast))
                if(temp == 0):
                    temp = 1                    
                f = float(131 * (contrast + 127))/temp
                alpha_c = f
                gamma_c = 127*(1-f)
                buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
            return buf

    def save_image(self, path, num ,image,ext):
        out_path = path + '/' + str(num) + ext
        # print("Saving Image" + out_path)
        cv2.imwrite(out_path, image)

IMAGES_PATH = "/Users/abdallaelshikh/Desktop/MIA/Coral/Coral-Detection/LabeledDataset/Images"
ANNOTATIONS_PATH = "/Users/abdallaelshikh/Desktop/MIA/Coral/Coral-Detection/LabeledDataset/Annotations"
aug = Augmentation(IMAGES_PATH, "/Users/abdallaelshikh/Desktop/MIA/Coral/Coral-Detection/LabeledDataset/Augmented_Images", ANNOTATIONS_PATH, "/Users/abdallaelshikh/Desktop/MIA/Coral-Detection/LabeledDataset/Augmented_Annotations", 3, True, True, True, True)
aug.augment(50,20,120)
