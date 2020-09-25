'''
Create train and val split dataset loaders
'''
import os
import imgaug as ia
import imgaug.augmenters as iaa
import random
import cv2
import numpy as np
import re
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, images_paths, labels, train_path, batch_size=32, num_classes=4, shuffle=False, augment=False, validate=False):
        self.labels = labels              # array of labels
        self.images_paths = images_paths        # array of image paths
        self.batch_size = batch_size          # batch size
        self.train_path = train_path
        self.num_classes = num_classes
        self.shuffle = shuffle             # shuffle bool
        self.augment = augment             # augment data bool
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        images = [cv2.cvtColor(cv2.imread(os.path.join(self.train_path, self.images_paths[k])), cv2.COLOR_BGR2RGB) for k in indexes]             
        
        # Preprocess and augment data
        if self.augment == True:
            images = self.augmentor(images)
            
        images = np.array(images)
        images = images.astype(np.float32)
        images = np.array([preprocess_input(img) for img in images])
        labels = [self.labels[k] for k in indexes]
        labels = to_categorical(labels, num_classes=self.num_classes, dtype='int8')
        return images, labels
    
    def augmentor(self, images):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        often = lambda aug: iaa.Sometimes(0.7, aug)

        seq = iaa.Sequential([            
            # Best Augmentation Strategy: Colour Augmentation
            often(
                iaa.WithChannels(0, iaa.Add((-30, 30))) # RGB = 0,1,2
                ),
            sometimes(
                iaa.LinearContrast((0.5, 2.0))
                ),
            sometimes(
                iaa.AddToBrightness((-30, 30))
                ),
            sometimes(
                iaa.GaussianBlur(sigma=(0,0.5))
                )
            
        ], random_order=True) # apply augmenters in random order
        
        return seq.augment_images(images)

def loadData(CLASSES, TRAIN_PATH, VAL_PATH, BATCH_SIZE):
    'Loads data into generator object'
    train_images_array = np.array([])
    train_labels_array = np.array([])
    val_images_array = np.array([])
    val_labels_array = np.array([])           
                
    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(CLASSES)):
        img_list = [f for f in os.listdir(os.path.join(TRAIN_PATH, CLASSES[category])) if ( re.match(r'^(?![\._]).*$', f) and f.endswith(".jpg") )] # filter out the apple files
        
        for i in range(len(img_list)):
            train_images_array = np.append(train_images_array, os.path.join(CLASSES[category], img_list[i]))
            train_labels_array = np.append(train_labels_array, category)
            
    for category in range(len(CLASSES)):
        img_list = [f for f in os.listdir(os.path.join(VAL_PATH, CLASSES[category])) if ( re.match(r'^(?![\._]).*$', f) and f.endswith(".jpg") )] # filter out the apple files
        
        for i in range(len(img_list)):
            val_images_array = np.append(val_images_array, os.path.join(CLASSES[category], img_list[i]))
            val_labels_array = np.append(val_labels_array, category)        
            
    # Use this if you want to dynamically split the train/validation sets:
    # train_images_array, val_images_array, train_labels_array, val_labels_array = train_test_split(train_images_array, train_labels_array, test_size=0.2, random_state=42)
            
    # Now create the generators
    train_data = DataGenerator(train_images_array, train_labels_array, TRAIN_PATH, batch_size=BATCH_SIZE, num_classes=len(CLASSES), augment=True, shuffle=True)
    val_data = DataGenerator(val_images_array, val_labels_array, VAL_PATH, batch_size=BATCH_SIZE, num_classes=len(CLASSES), augment=False, shuffle=True) 
    
    return train_data, val_data