import cv2
from helper import Segmentation
from keras.layers import Conv2D, Input, Dropout, MaxPooling2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, BatchNormalization
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
# from keras_segmentation.models.model_utils import get_segmentation_model
import numpy as np
from keras.callbacks import ModelCheckpoint
from config import *

def encoder_layer(inp, num_of_filters):
    """
    Encoder layers used in UNET Model
    """
    layer = Conv2D(num_of_filters, (3,3), padding = 'same', strides = 1, activation = 'relu')(inp)
    layer = Conv2D(num_of_filters, (3,3), padding = 'same', strides = 1, activation = 'relu')(layer)
    pool = MaxPooling2D((2,2))(layer)
    return layer, pool

def decoder_layer(inp, skip_layer, num_of_filters):
    """
    Decoder layers used in UNET Model
    """
    up = UpSampling2D((2,2))(inp)
    concatenated = Concatenate()([up, skip_layer])
    layer = Conv2D(num_of_filters, (3,3), padding = 'same', strides = 1, activation = 'relu')(concatenated)
    layer = Conv2D(num_of_filters, (3,3), padding = 'same', strides = 1, activation = 'relu')(layer)
    return layer

def bottleneck_layer(inp, num_of_filters):
    """
    Bottleneck layer of UNET Model
    """
    layer = Conv2D(num_of_filters, (3,3), padding = 'same', strides = 1, activation = 'relu')(inp)
    layer = Conv2D(num_of_filters, (3,3), padding = 'same', strides = 1, activation = 'relu')(layer)
    return layer

def UNET(width, height, color_depth, num_of_classes):
    #####################################ENCODER###################################
    input_layer = Input((width, height, color_depth))
    down_layer1, pool1 = encoder_layer(input_layer, 16)
    down_layer2, pool2 = encoder_layer(pool1, 32)
    down_layer3, pool3 = encoder_layer(pool2, 64)
    down_layer4, pool4 = encoder_layer(pool3, 128)
    #####################################BOTTLENECK################################
    bottleneck = bottleneck_layer(pool4, 256)
    #####################################DECODER###################################
    up_layer0 = BatchNormalization()(bottleneck)
    up_layer1 = decoder_layer(up_layer0, down_layer4, 128)
    up_layer2 = decoder_layer(up_layer1, down_layer3, 64)
    up_layer3 = decoder_layer(up_layer2, down_layer2, 32)
    up_layer4 = decoder_layer(up_layer3, down_layer1, 16)
    #####################################OUTPUT####################################
    output_layer = Conv2D(num_of_classes, (1,1), strides = 1, padding = 'same', activation = 'softmax')(up_layer4)
    model = Model(input_layer, output_layer)
    return model


segmentation = Segmentation()

#Loading train images
train_images = segmentation.load_images(TRAIN_IMAGES_PATH,"*.png", GRAYSCALE, True, (WIDTH, HEIGHT))

#Loading train  annotations
train_annotations = segmentation.load_images(TRAIN_ANNOTATIONS_PATH,"*.png", True, False, (WIDTH, HEIGHT))

#one hot encoding the annotations
one_hot_encoded_train_annotations = segmentation.get_segmentation_annotations(train_annotations, NUM_OF_CLASSES)

print("Train images shape: " + str(train_images.shape))
print("Train annotations shape: " + str(train_annotations.shape))
print("Train one hot encoded shape: " + str(one_hot_encoded_train_annotations.shape))

unet_model = UNET(WIDTH, HEIGHT, COLOR_DEPTH, NUM_OF_CLASSES)


if NUM_OF_CLASSES>2:
    unet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
else:
    unet_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

unet_model.summary()

unet_model.fit(
    train_images,
    one_hot_encoded_train_annotations,
    batch_size = BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
    )

unet_model.save(r"UNET20.h5")
print("Saved model to disk")