# MNIST Digit Classification with CNN

A convolutional neural network (CNN) implementation for classifying handwritten digits from the MNISTS dataset.


## Overview

This project implements a Deep Learning Model to recognise handwritten digits (0-9) using MNIST dataset. The model uses a CNN architecture to achieve high accuracy in digit classifcation.

## Dataset

The MNIST dataset contains 70.000 grayscale images of handwritten digits:

* Training set: 60.000 images
* Test size: 10.000 images
* Image size: 28x28 pixels
* Classes: 10 (digits 0-9)



## Model Architecture

The CNN consists of :
* Convolutional layers for feature extraction wiht 8 layers
* Max pooling layers for dimensionality reduction
* Dropout to prevent overfitting
* Fully connected layers for classification
* Output with 10 classes


## Requirements
```text
numpy==2.3.0
pandas==2.3.3
seaborn==0.13.2
matplotlib==3.10.7
torch==2.9.0
tqdm==4.67.1
torchvision==0.24.0
torchmetrics==1.8.2
gradio==6.0.1
```

## Installation

1. Clone the repository

```bash
git clone <repo>

```

2. Install dependencies

```bash
pip install -r requirements.txt
```

