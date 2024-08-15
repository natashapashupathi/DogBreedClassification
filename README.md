# Dog Breed Classification Project

This project focuses on building, training, and evaluating a deep learning model to classify different breeds of dogs based on images, there are a total of 120 breeds. The project uses TensorFlow and Keras for building the model and integrates various tools for data preprocessing, training, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [License](#license)

## Project Overview

The objective of this project is to create a machine learning model that can accurately classify dog breeds from images. The model is trained on a dataset of dog images labeled by breed and is evaluated based on its accuracy, precision, recall, and other metrics.

## Data

### Dataset

The dataset consists of images of dogs belonging to various breeds. The dataset is split into training and validation subsets for model training and evaluation.

- **Images Directory**: `train/`
- **Labels File**: `labels.csv`

### Data Preprocessing

Before training, the images are resized to 224x224 pixels and normalized by rescaling pixel values to the range [0, 1].

## Model Architecture

The model is a convolutional neural network (CNN) designed for image classification. The architecture includes:

- Convolutional layers with ReLU activation
- Max pooling layers
- Fully connected (dense) layers
- Dropout for regularization
- Softmax activation in the output layer for multi-class classification

## Training

The model is trained using the categorical cross-entropy loss function and the Adam optimizer. The training process involves the following steps:

1. **Data Augmentation**: Applied to increase the diversity of the training data.
2. **Model Compilation**: Configuring the model with the loss function, optimizer, and evaluation metrics.
3. **Training**: Running the training process with a set number of epochs.

## Evaluation

The model is evaluated on the validation dataset. The evaluation metrics include:

- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

### Accuracy

The accuracy of the model on the validation set is printed during the evaluation phase.

### Confusion Matrix

A confusion matrix is generated to visualize the model's performance across different classes.

### Classification Report

A detailed classification report is printed, showing precision, recall, and F1-score for each class.

## Usage

### Training the Model

```bash
python train.py

```
### Evaluating the Model

```bash
python evaluate.py
```

### Dependencies
Python 3.x
TensorFlow
Keras
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn

### Installing Dependencies
You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```
