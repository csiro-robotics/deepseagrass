# Repository for DeepSeagrass Dataset, Training Models and Original Paper.


The official repository for the paper: Multi-species Seagrass Detection and Classification from Underwater Images
\[[arXiv](https://arxiv.org/abs/2009.09924)]. 
This repository provides the steps and code to train and deploy deep learning models for detection and classification of multiple species of seagrass from underwater images.  It also contains our 'DeepSeagrass' dataset, codes and pre-trained models.

Our approach contributes to the field of automated seagrass detection by distinguishing between different types of seagrass and classifying seagrass on a per-patch basis.  This approach also provides location information about the seagrass in the frame without the need for dense pixel, polygon or bounding box labels.  If this repository contributes to your research, please consider citing the publication below.

```
Scarlett Raine and Ross Marchant and Peyman Moghadam and Frederic Maire and Brett Kettle and Brano Kusy (2020). 
Multi-species Seagrass Detection and Classification from Underwater Images. ACCEPTED TO DICTA 2020. 
```

### Bibtex
```
@misc{raine2020multispecies,
    title={Multi-species Seagrass Detection and Classification from Underwater Images},
    author={Scarlett Raine and Ross Marchant and Peyman Moghadam and Frederic Maire and Brett Kettle and Brano Kusy},
    year={2020},
    eprint={2009.09924},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
## Table of Contents
- [Installation](#installation)
- [DeepSeagrass](#deep-seagrass)
- [Models](#models)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)

<a name="installation"></a>
## Installation
We suggest using the Anaconda package manager to install dependencies.

1. Download Anaconda
2. Create a coda environment: conda create -n deepseagrass pip python=3.6
3. Activate the environment: conda activate deepseagrass
4. Install packages and libraries: pip install -r requirements.txt
5. Clone the adaptive learning rate scheduler and rolling buffer files from: https://github.com/microfossil/particle-classification/blob/master/miso/training/adaptive_learning_rate.py

<a name="deep-seagrass"></a>
## DeepSeagrass
Images were acquired across nine different seagrass beds in Moreton Bay, over four days during February 2020. Search locations were chosen according to distributions reported in the publicly available dataset. A biologist made a search of each area, snorkelling in approximately 1m of water during low to mid tide. In-situ search of seagrass beds resulted in 78 distinct geographic sub-areas, each containing one particular seagrass morphotype (or bare substrate).  Images were taken using a Sony Action Cam FDR-3000X from approximately 0.5m off the seafloor at an oblique angle of around 45 degrees. Over 12000 high-resolution (4624 x 2600 pixels) images were obtained. The images containing 'dense' seagrass were then processed into patches.  These patches were used to train our models.

The dataset is available for download from the CSIRO data portal at: https://doi.org/10.25919/spmy-5151.  Additional information about the dataset can be found in the accompanying file 'DeepSeagrass Dataset File' found at the same link.  The dataset contains necessary information and images for the 5-class classifier discussed in our paper, however it is not available at the 289 x 260 patch size.
 
![Dataset distinct seagrass](images/seagrass_map.png)
 
<a name="models"></a>
## Models
The best performing model reported in our paper is provided.  This model is trained to infer on image patches of 578 x 520 pixels. Our 578 x 520 pixel model achieved 98.2% to 99.6% precision and 98.0% to 99.7% recall for each class, and an overall accuracy of 98.8% on the validation dataset.  We achieved 88.2% overall accuracy on the DeepSeagrass unseen test set.  We found that there was an improvement in the accuracy when the 'Background' class was divided into 'Water' column and 'Substrate'.  We additionally provide a pre-trained model for this 5-class case. The 5-class model is only provided for 578 x 520 pixel patches. 

Additionally, we provide the best performing model for a patch size of 289 x 260 pixels.  This model can be used for lower resolution test images.

| Number of Classes | Image Patch Size | Accuracy | Model Link |
|-|-|-|-|
| 4 Classes | 578x520 pixels | 88.2% on test dataset | [Link](https://cloudstor.aarnet.edu.au/plus/s/nQ6JRNYvKaGqfaE?path=%2F520x578%20model) |
| 5 Classes | 578x520 pixels | 92.4% on test dataset | [Link](https://cloudstor.aarnet.edu.au/plus/s/nQ6JRNYvKaGqfaE?path=%2F5class_model) |
| | | |
| 4 Classes | 289x260 pixels | 94% on validation dataset | [Link](https://cloudstor.aarnet.edu.au/plus/s/nQ6JRNYvKaGqfaE?path=%2F260x289%20model) |

<a name="getting-started"></a>
## Getting Started
The approach takes the pre-trained weights of VGG16 on the ImageNet classification task.  The final Dense layers are removed and replaced with a decoder module trained on the DeepSeagrass dataset.  It is possible to deploy our trained model for inference or use our training script to train on a dataset of your own patches.  It is assumed that the images used are 4624 x 2600 pixels. 

Run the training script using:

```python train.py```

You can alter the number of classes and the batch size, for example:

```python train.py --batch_size=32 --num_classes=4```

A csv file is generated to store relevant class-specific metrics from training.  The model is saved as save.tf.
The script assumes that the training images are stored in the following file structure, where the folder names act as the image patch labels. 
```
    train
    ├── Strappy
    │   ├── Image0_Row1_Col0.jpg
    │   ├── Image0_Row1_Col1.jpg
    │   ├── Image0_Row1_Col2.jpg
    │   ├── .
    │   ├── .
    ├── Ferny
    │   ├── Image1_Row1_Col0.jpg
    │   ├── Image1_Row1_Col1.jpg
    │   ├── Image1_Row1_Col2.jpg
    │   ├── .
    │   ├── .
    ├── Rounded
    │   ├── Image50_Row1_Col0.jpg
    │   ├── Image50_Row1_Col1.jpg
    │   ├── Image50_Row1_Col2.jpg
    │   ├── .
    │   ├── .
    └── Background
        ├── Imag2679e_Row1_Col0.jpg
        ├── Image2679_Row1_Col1.jpg
        ├── Image2679_Row1_Col2.jpg
        ├── .
        └── .
 
    validate
    ├── Strappy
        ├── Image0_Row1_Col0.jpg
        ├── Image0_Row1_Col1.jpg
        ├── .
        └── .
    
    test
    ├── Strappy
        ├── Image0_Row1_Col0.jpg
        ├── Image0_Row1_Col1.jpg
        ├── .
        └── .
    etc.
```

### Inference / Evaluating the Model
The trained model can be reloaded and used on a test dataset of image patches using:

```python inference.py --num_classes=4 --patches``` 

This will automatically produce a class-wise confusion matrix and accuracy metrics.

If the user wants to save the incorrectly classified patches, then use:

```python inference.py --num_classes=4 --patches --save_incorrect```

When the patch is saved, the correct label and the inferred label are recorded in the name of the image.

If the script is going to be used for whole images, then use:

```python inference.py --num_classes=4 --whole_frame```

The script will infer on patches in the image.  The script saves the output image, with the inferred classes visualised as a colour mask on the original image.  Yellow is for the strappy class, blue is used for the rounded class, red is used for the ferny class and pink represents background.

For example:

![Output Image using visualise_inferences](images/output_image.jpg)

The file structure is the same as for training (above), except that test patches are stored in a folder called 'Test'.

A file for using the 289x260 pixel model for inference is also provided.  This script only works for the 4 class case, but uses the same flags as above.

<a name="acknowledgements"></a>
## Acknowledgements
This work was done in collaboration between CSIRO Data61, CSIRO Oceans and Atmosphere, Babel-sbf and QUT and was funded by CSIRO’s Active Integrated Matter and Machine Learning and Artificial Intelligence (MLAI) Future Science Platform. 
