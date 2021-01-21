# MNIST Digit Recognition
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/urastogi885/mnist-digit-recognition/blob/main/LICENSE)

## Overview
In this project, we perform hand-written digit recognition on the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
The dataset contains 60,000 training samples and 10,000 testing samples. Two experiments were conducted on this dataset.
Moreover, an SVM classifier with various kernels and a CNN classifier with 2 convolutional layers was designed to solve 
the classification problem.

## Dependencies
- [MATLAB](https://www.mathworks.com/products/matlab.html) (2019b version used)
- [Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning/index.html?s_tid=srchtitle)
- [Parallel Computing Toolbox](https://www.mathworks.com/help/parallel-computing/index.html?s_tid=CRUX_topnav) 
  (if you have GPU with 3.0 or higher compute power and want to run GPU accelerated training)
  
## Run
- Open the project folder in MATLAB and run `digit_recognition.mlx`
- `digit_recognition.mlx` provides the options for running the SVM and the CNN classifiers
- For SVM, you get the options to specify the dimensionality reduction (DR) method as well as the choice of kernel
- Note that the CNN takes about 2 hours (on GPU) to train and give out an accuracy
- Make sure you have the [Parallel Computing Toolbox](https://www.mathworks.com/help/parallel-computing/index.html?s_tid=CRUX_topnav) 
  installed on your system for MATLAB to access your GPU
- Note that the SVM classifier can take from 5 minutes to an hour depending on your choice of DR method and kernel
