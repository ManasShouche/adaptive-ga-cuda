Adaptive Genetic Algorithm for Network Intrusion Detection

This repository contains a C++/CUDA-based implementation of an adaptive Genetic Algorithm (GA). 
The project is designed to efficiently solve optimization problems by dynamically adjusting its parameters during runtime and exploiting the power of GPU parallelism.

To compile and run this code, you will need:

1)A C++ compiler (e.g., g++).
2)The NVIDIA CUDA Toolkit, including the nvcc compiler.
3)An NVIDIA GPU with a compatible driver.
4)Place all three source files (ga.h, ga_kernels.cu, main.cu) in the same directory.
5)Obtain the kdd_cup_train.csv and kdd_cup_test.csv dataset files and place them in the same directory. (See the Dataset section for more information).
->nvcc -o ga_program main.cu ga_kernels.cu
->./ga_program

Dataset
This project is built to work with the KDD Cup 1999 dataset, a standard benchmark for network intrusion detection. 
You will need to download dataset and use the Data_Preprocessing.ipynb to spilit into Train and Test sets.
Please ensure the files named kdd_cup_train.csv and kdd_cup_test.csv are placed in the same directory as the executable
