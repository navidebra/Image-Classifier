# Image Classifier App using Command Line

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, first a code for an image classifier was built with PyTorch, then convert it into a command line application.

### Project Overview

This project aims to use a given image database with chosen parameters using command line to train a neural network and predict accordingly using command line.

### Table of Contents

1. [Project Motivation](#project-motivation)
2. [Libraries](#libraries)
3. [File Descriptions](#files)

### Project Motivation
>This project is being done as a part of the Udacity AI Programming with Python Project. It involves developing an image classifier built with PyTorch, then converting it into a command line application.

### Libraries <a name="libraries"></a>
    PyTorch
    Numpy
    Matplotlib
    Seaborn
    Pillow

### File descriptions <a name="files"></a>

* Image Classifier Project.ipynb: Jupyter notebook containing main implementation and analysis.
* flowers/test: Folders containing images for testing.
* flowers/train: Folders containing images for training.
* flowers/valid: Folders containing images for validation.
* utils.py: File containing utility functions.
* train.py: File that can be used from Command Line to train a neural network with the given parameters.
    (sample: --data_dir /data --arch vgg16 --save_dir dir/output --learning_rate 0.01 --hidden_units 1024 --epochs 5 --gpu True)
* predict.py: File that can be used from Command Line to predict using the trained neural network with the given parameters.
    (sample: --image_path flower/1/img.jpg --checkpoint --checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu True)
