# Repository for Deep Seagrass dataset, training models and original paper.

# Multi-Seagrass
The official repository for the paper: Multi-species Seagrass Detection and Classification from Underwater Images

This repository provides the steps and code to train and deploy deep learning models for detection and classification of multiple species of seagrass from underwater images.  It also contains our 'DeepSeagrass' dataset, codes and pre-trained models.

## Paper
Our approach contributes to the field of automated seagrass detection by distinguishing between different types of seagrass and classifying seagrass on a per-patch basis.  This approach also provides location information about the seagrass in the frame without the need for dense pixel, polygon or bounding box labels.
```
 (2020). Multi-species seagrass detection and classification from underwater images. 
```

## DeepSeagrass
This dataset was collected over nine sites in Moreton Bay, Queensland, over four days in February 2020. A biologist snorkelled in approximately 1m of water during low to mid tide. Images were taken using a Sony Action Cam FDR-3000X from approximately 0.5m off the seafloor at an oblique angle of approximately 45 degrees. Over 12000 high-resolution (4624 x 2600 pixels) seagrass images were obtained, prioritising beds with only one seagrass type and ensuring to take images of bare substrate.  The images were sorted into their seagrass type and then divided into three categories (dense, medium, sparse) according to the density of seagrass present.  High-level morphological features (Strappy, Ferny and Rounded) were used to describe the type of seagrass in the frame. Additionally, a 'Mixed' class was created of images containing more than one type of seagrass and a 'Background' class was created to describe images without seagrass present to give a total of 11 classes in the dataset.  Images are labelled using folders.
 
## Preparing the Dataset
It is assumed that the images used are 4624 x 2600 pixels.  The dataset is first prepared by dividing each image into a grid of 5 rows and 8 columns - this yields patches of 578 x 520 pixels for training. The training and validation datasets are also created at this stage.
Fixed_Dataset_Generation.py

## Train the Model
The approach takes the pre-trained weights of VGG16 on the ImageNet classification task.  The final Dense layers are removed and replaced with a decoder module trained on the DeepSeagrass dataset.  It is possible to deploy our trained model for inference or use our training script to train on a dataset of your own patches.  We use the Adapative Learning Rate Scheduler from https://github.com/microfossil/particle-classification/blob/master/miso/training/adaptive_learning_rate.py.

# Setup
You will need to have Tensorflow and Keras installed.
