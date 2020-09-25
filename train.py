# Image classification script
# Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, Reshape
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import re
import argparse

from adaptive_learning_rate import AdaptiveLearningRateScheduler
from data_loader import loadData

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

'''
    This script is used for training but does not use kfold cross-validation.  The user can input
    different numbers for the number of classes and the batch
    size.
'''   
    
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
    VAL_PATH = 'Validate'
    
    NUM_ROWS = 5
    NUM_COLS = 8
    CROP_SHAPE = (int(2600//NUM_ROWS), int(4624//NUM_COLS), 3)
    
    train_data, val_data = loadData(CLASSES, TRAIN_PATH, VAL_PATH, BATCH_SIZE)
    
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
        
