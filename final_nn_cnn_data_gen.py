
# coding: utf-8
'''
data generation for cnn, nn
stratified shuffle split is used for test and val set split
'''
# In[1]:

import re
import pandas as pd
import numpy as np

#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils import shuffle

import pickle


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
                
        # integer columns and float columns summary (mean, median, min, max) 
        self.column_summary = {}
        for col in self.float_cols+self.ord_col_names:
            t_d = {}
            t_d['median'] = self.df[col].dropna().median()
            t_d['mean'] = self.df[col].dropna().mean()
            t_d['max'] = self.df[col].dropna().max()
            t_d['min'] = self.df[col].dropna().min()
            t_d['range'] = t_d['max'] - t_d['min']
            self.column_summary[col] = t_d
    
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


# In[5]:

#pipeline fn should be executed after creating train as it is used in that function
train = data('train.csv')


# In[4]:

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
        temp_data.df[col] = (temp_data.df[col] - train.column_summary[col]['min'])/(train.column_summary[col]['range']+0.00001)
    if dummy_var:
        temp_data.get_dummies()
    temp_data.sort_df()


# In[6]:

pipeline(train, dummy_var=True)


# In[10]:

X = train.df.drop(['id','target'], axis=1).as_matrix()
y = train.df['target'].as_matrix()


# In[8]:

#startified shuffle split will split data into train, test and validation with same ratio as
#target classes
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
a = sss.split(X, y)
for train_index, test_index in a:
    x_train, x_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index] 


# In[ ]:

print('Training shape: ', np.shape(x_train), np.shape(y_train))
print('Validation shape: ', np.shape(x_val), np.shape(y_val))


# In[ ]:

data = [(x_train,y_train),(x_val,y_val)]
with open('data_us.pickle', 'wb') as handle:
    pickle.dump(data, handle)

