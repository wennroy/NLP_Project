# CharCNN Implementation

In this folder, you can find our implementation of CharCNN. The datasets we use is a bit different. We divide the emotions into different groups according to the criterion mentioned in the paper (Positive, negative, ambiguous, neutral).

## Requirements

python==3.8

pytorch==1.10.2

sklearn==0.0

numpy==1.21.2

pandas==1.4.2

## Components

main.py: Run and train the model.

model.py: We define the CharCNN model. 

data_utils.py: Data preprocessing methods. 

eval.py: Evaluation helper methods.

## Instructions

To run the file, download them and ```python main.py```. If you do not want to run the file locally, you can have a look at the Jupyter Notebook. 