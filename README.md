# ML_food-recognition
# Food-11 Image Classification with ResNet50 & Recipe Suggestion System

This repository contains a complete end-to-end machine learning project for **food image classification** using the **Food-11 dataset** and **transfer learning with ResNet50**.  
All steps of the project from data loading to model training, evaluation, visualization, and recipe recommendations are implemented inside **one single notebook**, making the project easy to run and review.


## Project Overview

The goal of this project is to build a deep learning model capable of classifying images into **11 different food categories** using the Food-11 dataset.  
The project also includes a simple **recipe suggestion module** that recommends recipes based on the predicted food class.

The notebook performs:

- Loading and unzipping the Food-11 dataset  
- Automatically fixing directory structure for classes (0–10)  
- Data preprocessing and augmentation  
- Handling class imbalance with class weights  
- Building a ResNet50 transfer learning model  
- Training the top layers, then fine-tuning deeper layers  
- Evaluating performance on accuracy, loss, F1-scores, and confusion matrix  
- Generating plots for training curves  
- Displaying predictions on sample images  
- Suggesting recipes for the predicted category  

Everything is coded inside **one notebook file**.

## File Included


There are **no additional folders**, as the full implementation is consolidated in a single notebook.


## Dataset

This project uses the **Food-11 dataset**, which contains 16,643 images across 11 food categories.

The dataset must be manually downloaded from:

https://www.kaggle.com/datasets/vermaavi/food11

Then placed in your Google Drive under:


The notebook automatically unzips the file and organizes directories.


## Model Architecture (ResNet50)

The project uses **ResNet50 pretrained on ImageNet** (`include_top=False`).

Key components:

- Frozen convolutional base during initial training  
- Custom classifier head:
  - GlobalAveragePooling2D  
  - Dense(256, ReLU)  
  - Dropout(0.5)  
  - Softmax output  
- Fine-tuning the last **30 layers** of ResNet50  
- Optimizer: Adam  
- Loss: categorical crossentropy  


## Training Procedure

The training consists of two stages:

### **1. Head Training**
Only the custom dense layers are trained while the ResNet50 base remains frozen.

### **2. Fine-Tuning**
The last 30 layers of the base model are unfrozen and retrained using a smaller learning rate.

Callbacks used:

- `ModelCheckpoint`  
- `EarlyStopping`  
- `ReduceLROnPlateau`  


## Evaluation

The notebook generates:

- Training vs validation accuracy plot  
- Training vs validation loss plot  
- Confusion matrix  
- Classification report (precision, recall, F1-score)  
- Display of a sample prediction  
- Automatic recipe suggestions for the predicted class  

Achieved performance (example):

- **Test accuracy:** ~0.91  
- **Test loss:** ~0.28

## Recipe Suggestion Module

A simple Python dictionary (`RECIPE_DB`) maps each food class ID to a list of recipe objects.  
Based on the predicted class, the function:

```python
suggest_recipes(predicted_class_idx)
