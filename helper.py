import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
import os
import glob
import os.path as osp
import keras

class Segmentation():
    def load_images(self, files_path,ext, grayscale, normalize, target_size):
        """
        A funtion that takes the path of the images and returns a numpy array containing the images paths and a numpy array of one hot encoded labels
        """
        # data = load_files(files_path, shuffle = False) #load files
        # images = np.array(data['filenames']) #load images
        print("LOADING DATA")
        data = sorted(glob.glob(osp.join(files_path, ext)))
        images = np.array(data)  # load images
        return self.paths_to_tensor(images, normalize, target_size, grayscale)

    def get_segmentation_annotations(self, images, num_of_classes):
        return np_utils.to_categorical(images, num_of_classes)

    def path_to_tensor(self, img_path, normalize, target_size, grayscale):
        """
        A funtion that takes the path of the image and converts it into a 4d tensor to be fed to the CNN and normalizes them
        """
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, grayscale, target_size=target_size)
        # convert PIL.Image.Image type to 3D tensor with shape (32, 32, 3)
        x = image.img_to_array(img)
        # print('img shape: ', x.shape[:])
        # convert 3D tensor to 4D tensor with shape (1, 32, 32, 1) and return 4D tensor
        if(normalize):
            return np.expand_dims(x, axis=0)/255.0
        else:
            return np.expand_dims(x, axis=0)

    def paths_to_tensor(self, img_paths, normalize, target_size, grayscale):
        """
        A funtion that takes the path of the images and converts them into a 4d tensor to be fed to the CNN
        """
        print("CONVERTING IMAGES TO TENSORS")
        list_of_tensors = [self.path_to_tensor(img_path, normalize, target_size, grayscale) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors).astype('float32')   

    def load_npy(self, files_path, target_size):
        """
        A function that takes the path of npy files and loads them into a np array of tensors
        """
        path = files_path + "/*.npy"
        reshaped_size = target_size + (1,)
        files = sorted(glob.glob(path))
        out_size = (len(files), ) + reshaped_size 
        npy_arr = np.zeros(out_size)
        # print(npy_arr.shape)
        for num, file in enumerate(files):
            print("Generating npy from " + str(file))
            npy = np.load(file)
            print(npy[:50, :50])
            npy = np.resize(npy, target_size)
            npy = np.reshape(npy, reshaped_size)
            npy_arr[num] = npy
        return npy_arr


# class Augmentation():
#     def __init__(self, target_size, ):

# s = Segmentation()
# al, last = s.load_npy("/Users/abdallaelshikh/Desktop/MIA/Semantic/UNET_Semantic_Segmentation/semantic_segmentation/data_dataset_voc/SegmentationClass", (128,128))
