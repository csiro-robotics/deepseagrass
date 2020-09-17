# Image classification script
# Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# image augmentation
import imgaug as ia
import imgaug.augmenters as iaa

import random
import cv2
import re
import argparse

from adaptive_learning_rate import AdaptiveLearningRateScheduler

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

'''
    This script is used for training but does not use kfold cross-validation.  The user can input
    different numbers for the number of classes and the batch
    size.
'''

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, images_paths, labels, batch_size=32, shuffle=False, augment=False, validate=False):
        self.labels = labels              # array of labels
        self.images_paths = images_paths        # array of image paths
        self.batch_size = batch_size          # batch size
        self.shuffle = shuffle             # shuffle bool
        self.augment = augment             # augment data bool
        self.validate = validate            # validate bool
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images_paths) / BATCH_SIZE))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        
        indexes = self.indexes[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
        
        images = [cv2.cvtColor(cv2.imread(os.path.join(TRAIN_PATH, self.images_paths[k])), cv2.COLOR_BGR2RGB) for k in indexes]             
        
        # Preprocess and augment data
        if self.augment == True:
            images = self.augmentor(images)
            
        images = np.array(images)
        images = images.astype(np.float32)
        images = np.array([preprocess_input(img) for img in images])
        labels = [self.labels[k] for k in indexes]
        labels = to_categorical(labels, num_classes=NUM_CLASSES, dtype='int8')
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

def loadData():
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
    
    train_images, val_images, train_labels, val_labels = train_test_split(train_images_array, train_labels_array, test_size=0.2, random_state=42)
            
    # Now create the generators
    train_data = DataGenerator(train_images, train_labels, batch_size=BATCH_SIZE, augment=True, shuffle=True)
    val_data = DataGenerator(val_images, val_labels, batch_size=BATCH_SIZE, augment=False, shuffle=True, validate=True) 
    
    return train_data, val_data
    
    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='User specifies variables.')
    parser.add_argument('--batch_size', action="store", type=int, default=32)
    parser.add_argument('--num_classes', action="store", type=int, default=4)
    
    args = parser.parse_args()
                      
    EPOCHS = 10000
    NUM_CLASSES = args.num_classes
    IMAGE_BATCH = 1 * NUM_CLASSES  # One image per class
    BATCH_SIZE = args.batch_size
    CLASSES = ["Strappy", "Ferny", "Rounded", "Background"]
    TRAIN_PATH = 'Train'
    
    NUM_ROWS = 5
    NUM_COLS = 8
    CROP_SHAPE = (int(2600//NUM_ROWS), int(4624//NUM_COLS), 3)
    
    train_data, val_data = loadData()
    
    MODEL_PATH = "save.tf"
    CHECKPOINT_PATH = "model_weights.hdf5"
                                                
    # Decrease the learning rate
    learning_rate_reduction = AdaptiveLearningRateScheduler(nb_epochs=5,nb_drops=4,verbose=1)

    csv_logger = CSVLogger("training.csv", separator=',', append=False)
                
    # VGG16
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=CROP_SHAPE), pooling='avg')
        
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dropout(0.05)(headModel)
    headModel = Dense(512, activation='relu')(headModel)
    headModel = Dropout(0.15)(headModel)
    headModel = Dense(512, activation='relu')(headModel)
    headModel = Flatten(name="flatten2")(headModel)
    headModel = Dense(len(CLASSES), activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
        
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    # compile our model (this needs to be done after our setting our
    # layers to being non-trainable
    print("[INFO] compiling model...")
    opt = SGD(lr=1e-4, momentum=0.9)
    metrics_per_class = ["accuracy", tf.keras.metrics.Precision(class_id=0, name="precision_0"), tf.keras.metrics.Recall(class_id=0, name="recall_0"),
        tf.keras.metrics.Precision(class_id=1, name="precision_1"), tf.keras.metrics.Recall(class_id=1, name="recall_1"),
        tf.keras.metrics.Precision(class_id=2, name="precision_2"), tf.keras.metrics.Recall(class_id=2, name="recall_2"),
        tf.keras.metrics.Precision(class_id=3, name="precision_3"), tf.keras.metrics.Recall(class_id=3, name="recall_3")]
    model.compile(loss="categorical_crossentropy", optimizer='adam',
        metrics=metrics_per_class)
    model.summary() 
    
    print("[INFO] training the head...")
    # train on data
    history = model.fit_generator(generator=train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        steps_per_epoch=len(train_data) // BATCH_SIZE, 
        validation_steps=len(val_data) // BATCH_SIZE, 
        verbose=1,
        callbacks=[learning_rate_reduction, csv_logger]
        )        
    
    # Save the model to file
    print("[INFO] serializing network...")
    model.save(MODEL_PATH)

    # Save plot of training and validation accuracy and loss
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0].plot(history.history['loss'], label="TrainLoss")
    ax[0].plot(history.history['val_loss'], label="ValLoss")
    ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], label="TrainAcc")
    ax[1].plot(history.history['val_accuracy'], label="ValAcc")
    ax[1].legend(loc='best', shadow=True)
    plt.savefig("training_history.jpg") 
        