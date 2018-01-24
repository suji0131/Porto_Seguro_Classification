# Safe Driver Prediction - Porto Seguro ([Kaggle Competition](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction))

## Problem Description & Data Summary
Built a model (Wide & Deep Classifier) that predicts the probability that a driver will initiate an auto insurance claim in the next year. Each data point has ten continuous features, fourteen categorical features, seventeen binary variables and sixteen ordinal features. All features are anonymized and are given generic names. Target variable indicates whether the driver initiated a claim or not. So, this is a binary classification problem. But they are only interested in probablities, not class, indicated by the evaluation metric they chose, Normalized Gini Coefficient (NGC). NGC has a range of 0 (random guessing) to 1. There are 595,212 samples in the training set. Null values in the dataset are represented by -1.

## Data Preprocessing
As you can see from [1_Data_Pre_Processing](https://github.com/suji0131/Porto_Seguro_Classification/blob/master/1_Data_Pre_Processing.ipynb) file and image below the dataset is highly unbalanced. To address this, I initially used approaches like Synthetic Minority Over-sampling Technique (SMOTE) and Undersampling to balance the dataset and using balanced data to train classifier (decision trees, neural network, cnn(1-d convolutions) etc). These approches didn't yield the results I expected. I also built a Autoencoder but it also didn't improve the Gini coefficent. Explanation on why I finally chose the wide and deep network will be discussed later.

![Data Distribution](https://github.com/suji0131/Porto_Seguro_Classification/blob/master/Pre_processing_plots/Step%203.png)




