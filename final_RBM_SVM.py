
'''
RBM and SVM/SVC in pipeline
Parameter grid search for hyperparameter tuning
'''

# In[1]:

import pickle
import time
import math
import numpy as np

from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from imblearn.under_sampling import RandomUnderSampler 

# In[ ]:
strt_tym = time.time()
with open('data_us.pickle', mode = 'rb') as f:
  dataset = pickle.load(f)

x_train = dataset[0][0]
y_train = dataset[0][1]
print('x_train type: ', type(x_train))
print('x_train shape: ', np.shape(x_train))
print('y_train type and shape: ', type(y_train), ' ', np.shape(y_train))

x_val = dataset[1][0]
y_val = dataset[1][1] 
print('x_val type: ', type(x_val))
print('x_val shape: ', np.shape(x_val))
print('y_val type and shape: ', type(y_val), ' ', np.shape(y_val))


'''SVMs are effective in high dimensional spaces so no need to do below'''
'''
# In[2]:
with open('distribution.pickle', mode = 'rb') as g:
  data = pickle.load(f)= pickle.load(g)

mse0 = data[0]
mse1 = data[1]

i0 = np.mean(mse0, axis=0)
is0 = np.std(mse0, axis=0)

i1 = np.mean(mse1, axis=0)
is1 = np.std(mse1, axis=0)

idx = []
for i in range(len(i1)):
    if abs(i1[i] - i0[i]) > 0.01:
        idx.append(i)
        
print('cols being used: ', len(idx))

x_train = x_train[:, idx]
print('x_train type: ', type(x_train))
print('x_train shape: ', np.shape(x_train))
print('y_train type and shape: ', type(y_train), ' ', np.shape(y_train))

x_val = x_val[:, idx]
print('x_val type: ', type(x_val))
print('x_val shape: ', np.shape(x_val))
print('y_val type and shape: ', type(y_val), ' ', np.shape(y_val))
'''


# In[3]:
# Models we will use
sm = SVC(random_state=0, class_weight='balanced',cache_size=1000)
rbm = BernoulliRBM(random_state=0, verbose=False)

classifier = Pipeline(steps=[('rbm', rbm), ('sm', sm)])


param_grid = {'rbm__n_components': [50,100],
              'rbm__learning_rate': [0.06],
              'rbm__n_iter': [10,20],
              'sm__kernel': ['poly', 'rbf', 'sigmoid'],
              'sm__degree': [3,4]
              }

x_train, y_train = shuffle(x_train, y_train)

us = RandomUnderSampler(ratio={0:int(2*np.sum(y_train)), 1:np.sum(y_train)})
x_t, y_t = us.fit_sample(x_train, y_train)

# Grid-Search
grid = GridSearchCV(classifier, param_grid=param_grid,scoring='roc_auc', verbose=1)
grid.fit(x_t,y_t)

end_tym = time.time()
# In[ ]:

print('VM using RBM features:\n%s\n' % (metrics.classification_report(y_val, grid.predict(x_val))))
print('total 1s predicted: ', np.sum(grid.predict(x_val)))
print('Best parameter settings are:\n%s\n'%(grid.best_params_))
print('Time taken in mins: ', (end_tym-strt_tym)/60)