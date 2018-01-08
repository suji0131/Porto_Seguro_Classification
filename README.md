# Porto_Seguro_Classification
Kaggle Competition.
Built a Wide & Deep model that predicts the probability that a driver will initiate an auto insurance claim in the next year (Binary Classification).

1_Data_pre_processing.ipynb contains the initial data analysis
2_wide_n_deep_classifier.py is the submitted classifier

Bokeh plots of pre_processing are in Pre_processing_plots folder (Github strips javascript plots from jupyter notebook. So, they are added seperately)

Other files are either data generation files (last two words of the file will be data_gen.py) or other classifiers I tried.

Explored approaches:

wide_and_deep (submitted)

RBMs with Logistic Regression

RBMs with SVMs

SMOTE (with vanilla NN and CNNs)

Undersampling (with vanilla NN and CNNs)

Auto encoders
