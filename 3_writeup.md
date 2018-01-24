# Safe Driver Prediction - Porto Seguro ([Kaggle Competition](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction))

## Problem Description & Data Summary
Built a model (Wide & Deep Classifier) that predicts the probability a driver will initiate an auto insurance claim in the next year. Each data point has ten continuous features, fourteen categorical features, seventeen binary variables and sixteen ordinal features. All features are anonymized and are given generic names. Target variable indicates whether the driver initiated a claim or not. So, this is a binary classification problem. But they are only interested in probablities, not class, indicated by the evaluation metric they chose, Normalized Gini Coefficient (NGC). NGC has a range of 0 (random guessing) to 1. There are 595,212 samples in the training set. Null values in the dataset are represented by -1.

## Data Preprocessing
As you can see from [1_Data_Pre_Processing](https://github.com/suji0131/Porto_Seguro_Classification/blob/master/1_Data_Pre_Processing.ipynb) file and image below the dataset is highly unbalanced. To address this, I initially used approaches like Synthetic Minority Over-sampling Technique (SMOTE) and Undersampling to balance the dataset and using balanced data to train classifier (decision trees, neural network, cnn(1-d convolutions) etc). These approches didn't yield the results I expected. I also built a Autoencoder but it also didn't improve the Gini coefficent. Explanation on why I finally chose the wide and deep network will be discussed later.

![Data Distribution](https://github.com/suji0131/Porto_Seguro_Classification/blob/master/Pre_processing_plots/Step%203.png)

Some of the columns are completely dominated by null values (> 95%) which are dropped. And normality test on continuous and ordinal columns showed that they don't follow normal distribution. So, null values for those  columns are replaced by median values instead of mean values. There is no significant correlation and multicorrelation between the features (which is why CNN are a bad idea for this problem). For discreet variables, distribution of classes is studied and features with a single dominating class are removed from dataset (all plots are in [Pre_processing_plots](https://github.com/suji0131/Porto_Seguro_Classification/tree/master/Pre_processing_plots) folder). Using Random Forests (optimal parameters are found using grid search) I mapped feature importance, shown below. I dropped features with low importance and trained wide & deep model with subset of important features but results were worse than using whole data, it goes to show that more data (in this case features) you have at model's disposal better the results. 

![Feature Importance](https://github.com/suji0131/Porto_Seguro_Classification/blob/master/Pre_processing_plots/Feature%20Importance%20Plot.png)

## Model Architecture
Here is an excellent [article](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) on wide & deep model from people who developed it and here is their [research paper](https://arxiv.org/abs/1606.07792). This model is developed as app recommendation system for android playstore. Wide models (like logistic regression) are good at memorization, learning the frequent co-occurrence of items or features and exploiting the correlation available in the historical data. Deep models (neural networks) are really good at generalization,  relevant feature combinations that have never or rarely occurred in the past. Wide & Deep model combines the power of both to give balanced recommendations. 

There are similarities with data we are working on and their data, like sparse feature vectors and over generalization resulting from unbalanced data. So, I decided to give it a go. All sparse features are included in the wide part and others are used in deep part. Crossed features are obtained using features used in wide part. I read on tensorflow blog that in general hashing and bucketizing improves the accuracy(even the author said they don't know why but I'm guessing hashing somehow improves the differentiation between classes of a feature, I may be wrong). 
```
#crossed columns, hashing
crossed_columns = []
for i in itertools.combinations(base_col_names, 2):
    j = i[0]
    k = i[1]
    a = train.column_summary[j]['unq'] * train.column_summary[k]['unq']
    crossed_columns.append(tf.feature_column.crossed_column([j,k], hash_bucket_size=int(round(math.log(a, 2), 0)))) 
```

Out of all classifiers I tried this one gave me best normalized gini coefficient, 0.281 (competition winner has a score of 0.297). Model was trained on Google Cloud using 1 NVIDIA Tesla K80 GPU and 2 vCPUs (10 GB memory). Training time was nearly half an hour for 5 epochs with batch size of 9000. Deep part of the model has three hidden layers with 100, 150 and 50 neurons respectively. 
```
#estimator
estimator = tf.estimator.DNNLinearCombinedClassifier(model_dir='wide_n_deep1',
    linear_feature_columns=base_columns + crossed_columns,
    dnn_feature_columns=deep_cols, dnn_hidden_units=hid_units)
```
