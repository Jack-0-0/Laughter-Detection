# Evaluation
evaluation.ipynb calculates the performance metrics and plots the ROCs for each language test set. 

## Requirements
The following packages are needed for evaluation:
* Python 3.7.0
* numpy 1.17.2
* matplotlib 3.1.1
* scikit-learn 0.21.3
* jupyter 1.0.0
* tensorflow-gpu 2.0.0
* cuda 10.0-cudnn7.5.1
* Keras 2.3.0

## Datasets and Trained Models
The evaluation notebook needs a trained model, the training data used to create the model and the three language test sets. These can be downloaded from [link to be added].

Alternatively, the datasets can be created as described in the preprocessing section [here](/preprocessing). The trained model can be created by following the steps in the train section [here](/train).

## Usage
At the top of the notebook:
* set the path to the model for evaluation
* set the path for the training data
* set paths to the test data
* set the number of mels (set this to 64 for 22,050 Hz experiments or 45 for 8,000 Hz experiments)

Then run each code block in order. This will print the TPR, FPR, FNR, TNR and F1 scores and plot the ROC for each of the language test sets.
