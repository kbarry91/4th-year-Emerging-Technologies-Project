<p align="center">
  <img src = "https://i.imgur.com/wsqbmaa.png/">
</p>

# Emerging Technologies Project
## Motivation

This repository was developed as a project for the module Emerging Technologies as part of a Software Developement Degree in GMIT. The project involves writing documentation, code, and comments in the programming language Python and using the Jupyter notebook software. The purpose of this project was to engage in new technologies and machine learning techniques.


## Introduction
The project contains 5 main folders. Folder 1,2,3,5 are Jupyter notebooks and folder 4 is Python script.
1. NumPy Random Notebook
2. Iris dataset notebook
3. MNIST dataset notebook
4. MNIST digit Recognition Script
5. MNIST Digit Recognition Notebook


## Prerequisities
In order to debug this Application you must have **Anaconda** installed. Anaconda is a popular Python data science platform
[How to install Anaconda](https://conda.io/docs/user-guide/install/windows.html) Make sure to add Anaconda to your system path.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Downloading the project

#### Using the zip file
- Go to (https://github.com/kbarry91/Emerging-Technologies.git)
- Select *Download Zip*
- Extract the full folder in the target directory

#### Alternatively using GIT
- Ensure Git is installed https://git-scm.com/downloads
- Navigate to target directory of project in CMD
>Git clone https://github.com/kbarry91/Emerging-Technologies.git

### Install ther required libraries 
- navigate to the project directory
> $ conda install keras
> $ conda install numpy
> $ conda install scikit-learn
> $ conda install matplotlib 
> $ conda install opencv-python

## Run The project

### Run The Jupyter notebooks
- Navigate to the project directory and enter following command.
> $ jupyter notebook
- jupyter notebook will launch in the browser.
  
### Running the Python script
- Navigate to the script directory and enter following command.
> python digitrec.py  
- The program will run in the command window.

## Contents
### NumPy Random Notebook
This jupyter notebook explains the concepts behind the use of the numpy random package, including plots and various distributions.NumPy is the fundamental package for scientific computing with Python. It contains manu useful functions for dealing with data such as:
- N-dimensional array objects
- Broadcasting functions
- Linear algebra
- Fourier transform
- Random number capabilities
  
The NumPy library can be used as an efficient multi-dimensional container of generic data. Throughout this project NumPy was used when manipulating datasets.

### Iris Dataset Notebook
The Iris dataset notebook takes a look at the Iris dataset which is now widely used as a data set for testing purposes in computer science. Many developers consider the classification of the iris dataset as the Hello World of Machine Learning.

The Iris flower dataset is a specific set of information compiled by Ronald Fisher, a biologist, in the 1930s. It describes particular biological characteristics of various types of Iris flowers, specifically, the length and width of both pedals and the sepals, which are part of the flower’s reproductive system.

This notebook will demonstrate what is known as Supervised Learning using label data as we are trying to learn the relationship between the data (Iris measurements) and the outcome which is the species of Iris. This is in contrast to Unsupervised Learning with unlabeled data where we would only have the measurement data but not the species

### MNIST Dataset Notebook
The MNIST dataset was constructed from two datasets of the US National Institute of Standards and Technology (NIST). The training set consists of handwritten digits from 250 different people, 50 percent high school students, and 50 percent employees from the Census Bureau. Note that the test set contains handwritten digits from different people following the same split.

The MNIST dataset contains 60,000 training images and 10,000 test images along with their corresponding labels.

In this notebook we will look at how to read the MNIST dataset efficiently into memory.

### MNIST Digit Recognition Script
A Python script that takes an image ﬁle containing a handwritten digit and identiﬁes the digit using a supervised learning algorithm and the MNIST dataset. The program runs in the command line and builds a neural network. The neural network can then predict the value of a digit from a user inputed image.

### MNIST Digit Recognition Notebook
This function of this Jupyter notebook is to explain how the python script [Source link to script](https://github.com/kbarry91/Emerging-Technologies/tree/master/4-MNIST%20Digit%20Recognition%20Script) works and to discuss its performance.".

## Research 
In order to develope this application alot of effort went into research as UWP was a new platform to me. The microsoft docs available at https://docs.microsoft.com/en-us/windows/uwp/ provided alot of insight as to how this app could be developed. Any code adapted from external sources has been clearly referenced through the code files.


## Authors
* **Kevin Barry** - *Initial work* - [kbarry91](https://github.com/kbarry91)

## Acknowledgments And References
* Lecturer [Ian McLoughlin](https://github.com/ianmcloughlin/) of GMIT 
* All references and resources used throughout the development process are available in each notebook.
