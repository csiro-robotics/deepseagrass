from __future__ import absolute_import, division, print_function, unicode_literals
import os
import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# image augmentation
import imgaug as ia
import imgaug.augmenters as iaa

import random
import time
import cv2
import re
import argparse

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, images_paths, labels, batch_size=1, image_dimensions = (2600, 4624, 3), shuffle=False, augment=False):
        self.labels       = labels              # array of labels
        self.images_paths = images_paths        # array of image paths
        self.dim          = image_dimensions    # image dimensions
        self.batch_size   = batch_size          # batch size
        self.shuffle      = shuffle             # shuffle bool
        self.augment      = augment             # augment data bool
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
        # selects indices of data for next batch (chooses an image using 'index' provided by shuffling the dataset at the end of each pass)
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]        
        
        images = [cv2.cvtColor(cv2.imread(os.path.join(TEST_PATH, self.images_paths[k])), cv2.COLOR_BGR2RGB) for k in indexes]
        images = np.array(images)
        images = images.astype(np.float32)
        images = np.array([preprocess_input(img) for img in images])
        
        labels = [self.labels[k] for k in indexes]
        labels = to_categorical(labels, num_classes=NUM_CLASSES, dtype='int8')
        
        return images, labels

def loadData():
    'Loads data into generator object'
    test_images_array = []
    test_labels_array = []
    
    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(CLASSES)):
        img_list = [f for f in os.listdir(os.path.join(TEST_PATH, CLASSES[category])) if ( re.match(r'^(?![\._]).*$', f) and f.endswith(".jpg") )] # filter out the apple files
        num_img = len(img_list)
        
        for i in range(num_img):
            test_images_array.append(os.path.join(CLASSES[category], img_list[i]))
            test_labels_array.append(category)
                 
    test_data = DataGenerator(np.asarray(test_images_array), np.asarray(test_labels_array), batch_size=1, augment=False, shuffle=False)
    return test_data, test_images_array  
	
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='User specifies variables.')
    parser.add_argument('--patches', action="store_true", default=False)
    parser.add_argument('--save_incorrect', action="store_true", default=False)
    parser.add_argument('--whole_frame', action="store_true", default=False)

    args = parser.parse_args()

    NUM_CLASSES = 4
    CLASSES = ["Strappy", "Ferny", "Rounded", "Background"]
    TEST_PATH = 'Test'
    
    NUM_ROWS = 10
    NUM_COLS = 16
    NUM_CELLS = NUM_ROWS*NUM_COLS
    CROP_SHAPE = (int(2600//NUM_ROWS), int(4624//NUM_COLS), 3)
    
    MODEL_PATH = "save.tf"
    
    # load the trained model from the .pb file
    new_model = load_model(MODEL_PATH)
    new_model.build((None, CROP_SHAPE[0], CROP_SHAPE[1], CROP_SHAPE[2]))
    
    if args.patches == True:
        test_data, test_images_array = loadData()
        
        # Predicting the classes for each of the cells in the test images
        print("[INFO] evaluating trained model...")
        predIdxs = new_model.predict_generator(test_data)
        predIdxs = np.argmax(predIdxs, axis=1)
        
        labels = test_data.labels
        
        # Calculate metrics and confusion matrices for the test images
        print(classification_report(labels, predIdxs, target_names=CLASSES))
        print(confusion_matrix(labels, predIdxs))
        print("Normalized Confusion Matrix")
        C = confusion_matrix(labels, predIdxs)
        print( C / C.astype(np.float).sum(axis=1, keepdims=True) )
        
        # Save incorrect image patches with details of the inferred label and actual label
        if args.save_incorrect == True:  
            for i in range(len(predIdxs)):
                # Check if the prediction matches the actual label
                if predIdxs[i] == test_data.labels[i]:
                    continue
                image = cv2.imread(os.path.join(TEST_PATH, test_images_array[i]))
                save_path = "Actual_"+str(test_data.labels[i])+"_Inference_"+str(predIdxs[i])+"_no"+str(i)+".jpg"
                cv2.imwrite(save_path, image)  

    # Save images with an overlay colour mask to visualise the inferences
    if args.whole_frame == True:
        print("[INFO] generating output images...")
        test_data, test_images_array = loadData()
        
        predIdxs = new_model.predict_generator(test_data)
        predIdxs = np.argmax(predIdxs, axis=1)

        # Find the number of individual test images
        num_images = int(len(predIdxs)/NUM_CELLS)

        # Split the prediction classes into individual lists corresponding to the images
        predIdxs_split = [predIdxs[i:i + NUM_CELLS] for i in range(0, len(predIdxs), NUM_CELLS)]

        fig = plt.figure(figsize = (10,10), dpi = 400)   

        for i in range(num_images):
            print("[INFO] building output image no {}...".format(i))
            inferences = predIdxs_split[i]

            fig , ax = plt.subplots(num_images,1,figsize = (10,10), dpi = 400)
            ax = fig.add_subplot(num_images,1,i+1)

            image = cv2.imread(os.path.join(TEST_PATH, test_images_array[i]))
            image_np = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            overlay = image_np.copy()
            output = image_np.copy()

            # how transparent the coloured mask should appear
            alpha = 0.5

            # These colours correspond to the matplotlib display!
            red = (255, 0, 0)
            blue = (0, 0, 255)
            pink = (255,20,147)
            orange = (255,165,0)

            # For each image, reset the cell index back to zero
            index = 0

            for y in range(0,NUM_ROWS):
                for x in range (0,NUM_COLS):
                    x1 = x * CROP_SHAPE[1]
                    y1 = y * CROP_SHAPE[0]

                    x2 = (x+1) * CROP_SHAPE[1]
                    y2 = (y+1) * CROP_SHAPE[0]

                    if inferences[index] == 0:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), orange, -1) # Strappy
                    elif inferences[index] == 1:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), red, -1) # Ferny
                    elif inferences[index] == 2:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), blue, -1) # Round
                    elif inferences[index] == 3:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), pink, -1) # Background

                    # Increment the cell index by 1 each time to obtain the next inference in the array
                    index = index + 1

            # apply the overlay
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            ax.imshow(output)

            # save the output images
            save_path = "Label_"+str(test_data.labels[i])+"_no"+str(i)+".jpg"
            cv2.imwrite(save_path, cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
        
print("[INFO] finished inference")
