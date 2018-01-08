
# coding: utf-8

# https://www.tensorflow.org/tutorials/wide_and_deep
# 
# https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket
# 
# https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
# 
# https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
# 
# https://github.com/random-forests/tensorflow-workshop/blob/master/examples/07_structured_data.ipynb

# In[1]:
import os
import re
import math
import time
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score
#from sklearn.utils import shuffle

import pickle

print('Importing libraries --done')
#tf.reset_default_graph()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

epochs_ = 1
bat_size = 9000
hid_units = [100,150,50]

# In[2]:

# sorts strings properly
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


# In[3]:

'''
data class has the dataframe, keeps track of its bin, cat, ord etc. col names and has summary of cont. and ord. cols
'''
class data:
    '''
    filename: path of csv file to be read
    '''
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        
        #don't use cols_ anywhere except in init
        self.cols_ = self.df.columns.tolist()
        
        self.float_cols = self.df.select_dtypes(include=['float64']).columns.tolist()
        #cat and bin column names
        self.cat_col_names = [col for col in self.cols_ if '_cat' in col]
        self.bin_col_names = [col for col in self.cols_ if '_bin' in col]

        self.ord_col_names = []
        for col in self.cols_:
            if ('_cat' in col) or ('_bin' in col) or (col in ['id', 'target']) or (col in self.float_cols):
                a = 1
            else:
                self.ord_col_names.append(col)
                
        self.get_summary()
                    
    '''
    removes the column from dataframe and its name from appropriate list
    col_name_list_: is a list of columns to be removed
    '''            
    def remove_cols(self, col_name_list_):
        for col_name in col_name_list_:  
            self.df.drop([col_name],axis=1, inplace = True)
            if col_name in self.cat_col_names:
                self.cat_col_names.remove(col_name)
            elif col_name in self.float_cols:
                self.float_cols.remove(col_name)
            elif col_name in self.bin_col_names:
                self.bin_col_names.remove(col_name)
            elif col_name in self.ord_col_names:
                self.ord_col_names.remove(col_name)
                
    def sort_df(self):
        ordered_cols = natural_sort(self.df.columns.tolist())
        self.df = self.df[ordered_cols]
        
    def get_dummies(self):
        self.df = pd.get_dummies(self.df,columns=self.cat_col_names,prefix=self.cat_col_names)
        
    def get_summary(self):
        # integer columns and float columns summary (mean, median, min, max) 
        self.column_summary = {}
        for col in self.float_cols+self.ord_col_names:
            t_d = {}
            t_d['median'] = self.df[col].dropna().median()
            t_d['max'] = self.df[col].dropna().max()
            t_d['min'] = self.df[col].dropna().min()
            self.column_summary[col] = t_d
        for col in self.cat_col_names+self.bin_col_names:
            t_d = {}
            #freq ocurrening value
            t_d['freq'] = self.df[col].value_counts().idxmax()
            t_d['unq'] = len(self.df[col].unique().tolist())
            self.column_summary[col] = t_d


# In[4]:

print('Creating training class')
#pipeline fn should be executed after creating train as it is used in that function
train = data('train.csv')


# In[5]:

'''
temp_data: it should be a data object
'''
def pipeline(temp_data, dummy_var=False):
    drop_cols = ['ps_car_03_cat','ps_car_05_cat', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin','ps_ind_13_bin', 
                'ps_car_10_cat', 'ps_ind_14']
    temp_data.remove_cols(drop_cols)
    
    '''
    I'm treating ord cols as cont. variables for this problem.
    replace null values with median and makes the column mean 0 and every value bt -1 and +1
    '''
    for col in temp_data.float_cols + temp_data.ord_col_names:
        #replacing nan values with median for float and integer columns
        #here for this problem -1 means nan
        #for categorical variables we will make all the dummy variables zero
        
        temp_data.df[col].replace(-1, train.column_summary[col]['median'],inplace=True)
        temp_data.get_summary()
        #temp_data.df[col] = (temp_data.df[col] - train.column_summary[col]['min'])/(train.column_summary[col]['range']+0.0001)
        
    for col in temp_data.bin_col_names + temp_data.cat_col_names:
        #replacing nan values with most frequently occuring value for bin and cat columns
        #most_freq = temp_data.df[col].value_counts().idxmax()
        temp_data.df[col].replace(-1, train.column_summary[col]['freq'],inplace=True)
        
    if dummy_var:
        temp_data.get_dummies()
    temp_data.sort_df()


# In[6]:

print('applying pipeline')
pipeline(train, dummy_var=False)

test = data('test.csv', train_file = False)
print('applying pipeline to test file')
pipeline(test, dummy_var=False)

# In[7]:

#ensure no nulls -1 in the dataset
#train.df[train.df == -1].isnull().sum()


# In[8]:

#shuffling the rows of a dataframe
train.df = train.df.sample(frac=1).reset_index(drop=True)


# In[9]:

y = train.df.pop('target')


# In[10]:

print('splitting df')
df_train, df_test, y_train, y_test = train_test_split(train.df, y, test_size=0.15, random_state=42)


# In[11]:

def create_train_input_fn(): 
    return tf.estimator.inputs.pandas_input_fn(
        x=df_train,
        y=y_train, 
        batch_size=bat_size,
        num_epochs=epochs_, # Repeat forever
        shuffle=True)

def create_test_input_fn():
    return tf.estimator.inputs.pandas_input_fn(
        x=df_test,
        y=y_test, 
        num_epochs=1, # Just one epoch
        shuffle=False) # Don't shuffle so we can compare to census_test_labels later

test_id = test.df.pop('id')
def create_test_sub_fn():
    return tf.estimator.inputs.pandas_input_fn(
        x=test.df,
        y=test_id, 
        num_epochs=1, # Just one epoch
        shuffle=False) # Don't shuffle so we can compare to census_test_labels later
print('train, test inp fns created')


# In[12]:

#defining base feature columns
print('defining base feature columns')
'''
vars()['string_name'] creates a variable with the name string_name
'''
#continuous cols 
for col in train.float_cols:
    a = train.column_summary[col]['min']
    b = train.column_summary[col]['max'] - train.column_summary[col]['min']
    vars()[col] = tf.feature_column.numeric_column(col, normalizer_fn=lambda x:(x - a)/b)

#binary cols
for col in train.bin_col_names:
    vars()[col] = tf.feature_column.categorical_column_with_identity(key=col,num_buckets=2,
                                                                     default_value=train.column_summary[col]['freq'])
    
#categorical cols
for col in train.cat_col_names:
    #number of levels
    a = train.column_summary[col]['unq']
    vars()[col] = tf.feature_column.categorical_column_with_identity(key=col,num_buckets=a+1,
                                                                     default_value=train.column_summary[col]['freq'])
    '''
    #earlier idea
    if a == 2:
        vars()[col] = tf.feature_column.categorical_column_with_identity(key=col,num_buckets=a,
                                                                     default_value=train.column_summary[col]['freq'])
    else:
        # number of hash buckets to use
        no_hash_buc = int(round(math.log(a, 2), 0)) #log n to the base 2, where n is no of levels
        vars()[col] = tf.feature_column.categorical_column_with_hash_bucket(key=col,hash_bucket_size=no_hash_buc)
    '''

#some ord cols are bucketized and some are fed as cont. cols
deep_ord_buc_cols = ['ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10',
                'ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14']

for col in train.ord_col_names:
    if col in deep_ord_buc_cols:
        vars()[col] = tf.feature_column.numeric_column(col)
    else:
        a = train.column_summary[col]['min']
        b = train.column_summary[col]['max'] - train.column_summary[col]['min']
        vars()[col] = tf.feature_column.numeric_column(col, normalizer_fn=lambda x:(x - a)/b)

#bucketized ord cols
ps_calc_06_buc = tf.feature_column.bucketized_column(ps_calc_06, [5,6,7,8,9,10])
ps_calc_07_buc = tf.feature_column.bucketized_column(ps_calc_07, [1,2,3,4,5,6,7])
ps_calc_08_buc = tf.feature_column.bucketized_column(ps_calc_08, [6,7,8,9,10,11,12])
ps_calc_09_buc = tf.feature_column.bucketized_column(ps_calc_09, [1,2,3,4,5])
ps_calc_10_buc = tf.feature_column.bucketized_column(ps_calc_10, [i+3 for i in range(12)])
ps_calc_11_buc = tf.feature_column.bucketized_column(ps_calc_11, [i+2 for i in range(10)])
ps_calc_12_buc = tf.feature_column.bucketized_column(ps_calc_12, [i+1 for i in range(4)])
ps_calc_13_buc = tf.feature_column.bucketized_column(ps_calc_13, [i+1 for i in range(7)])
ps_calc_14_buc = tf.feature_column.bucketized_column(ps_calc_14, [i+3 for i in range(12)])


# In[13]:

print(train.ord_col_names)


# In[14]:

print('features for wide part')
#features for wide part
#sparse base cols
sparse_bin = ['ps_ind_08_bin','ps_ind_09_bin','ps_ind_17_bin','ps_ind_18_bin','ps_calc_15_bin','ps_calc_20_bin']
base_col_names =  sparse_bin + train.cat_col_names

base_columns = []
for i in base_col_names:
    base_columns.append(vars()[i])

#crossed cols of base cols
crossed_columns = []
for i in itertools.combinations(base_col_names, 2):
    j = i[0]
    k = i[1]
    a = train.column_summary[j]['unq'] * train.column_summary[k]['unq']
    crossed_columns.append(tf.feature_column.crossed_column([j,k], hash_bucket_size=int(round(math.log(a, 2), 0)))) 


# In[15]:

deep_bin_cols = [x for x in train.bin_col_names if x not in sparse_bin]
deep_ord_cont_cols = [x for x in train.ord_col_names if x not in deep_ord_buc_cols]


# In[16]:

print('features for deep part')
deep_cols = []
for i in train.float_cols+deep_ord_cont_cols:
    deep_cols.append(vars()[i])

for i in deep_bin_cols:
    deep_cols.append(tf.feature_column.indicator_column(vars()[i]))
    
for i in deep_ord_buc_cols:
    deep_cols.append(tf.feature_column.indicator_column(vars()[i+'_buc']))


# In[17]:
start_tym = time.time()
print('training....')
train_input_fn = create_train_input_fn()

estimator = tf.estimator.DNNLinearCombinedClassifier(model_dir='wide_n_deep1',
    linear_feature_columns=base_columns + crossed_columns,
    dnn_feature_columns=deep_cols, dnn_hidden_units=hid_units)

estimator.train(train_input_fn)
print('training -- done')
end_tym = time.time()

print('Training time in mins: ',(end_tym-start_tym)/60)

# In[18]:

#print('Evaluating on test set...')
#test_input_fn = create_test_input_fn()
#print(estimator.evaluate(test_input_fn))


# In[25]:

# reinitialize the input function
print('predicting...')
test_input_fn = create_test_input_fn()
predictions = estimator.predict(test_input_fn)


# In[26]:

y_test_temp = y_test.tolist()
i = 0
pred_l = []
pred_prob = []
for prediction in predictions:
    '''
    prediction is a dictionary with keys
    '''
    true_label = y_test_temp[i]
    predicted_label = prediction['class_ids'][0]
    pred_prob.append(prediction['probabilities'][1])
    pred_l.append(predicted_label)
    # Uncomment the following line to see probabilities for individual classes
    #print(prediction) 
    #print("Example %d. Actual: %d, Predicted: %d" % (i, true_label, predicted_label))
    i += 1


# In[24]:

print('Precision: ',precision_score(y_test_temp,pred_l))
print('Recall: ', recall_score(y_test_temp,pred_l))
print('Accuracy: ', accuracy_score(y_test_temp,pred_l))


# In[29]:

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
 
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


# In[30]:

print('Normalized Gini co-effcient: ', gini_normalized(y_test_temp,pred_prob))


print('predicting submission file...')
test_sub_fn = create_test_sub_fn()
predictions_sub = estimator.predict(test_sub_fn)

id_ = test_id.tolist()
i = 0
pred_prob_s = []
for prediction1 in predictions_sub:
    '''
    prediction is a dictionary with keys
    '''
    pred_prob_s.append(prediction1['probabilities'][1])
    i += 1
    
final_dict = {'id': id_,
             'target': pred_prob_s}

final_df = pd.DataFrame.from_dict(final_dict, orient='columns')
final_df.to_csv('submission.csv', index=False)





