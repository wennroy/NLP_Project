# biLSTM with Single Label

At first, we assume that each text only has one label. We built the model in PyTorch and results were not good. 

## Requirement

python==3.8

pytorch==1.10.2

sklearn==0.0

numpy==1.21.2

pandas==1.4.2

## Components

utils.py: It stores label mapping functions and vocabulary dictionary class.

main.py: Run and train the model.

model.py: We define the model in this file.

preprocess.py: It includes the data loader and sentence padding method that we use for preprocessing the data.

## Instructions

To run the file, download them and ```python main.py```.