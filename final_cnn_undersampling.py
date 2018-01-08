import pickle
import time
import numpy as np
import tensorflow as tf

from collections import Counter

from sklearn.utils import shuffle

from imblearn.under_sampling import RandomUnderSampler 

from tensorflow.contrib.layers import flatten

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

# Number of training examples
n_train = len(y_train)

# Number of validation examples
n_validation = len(y_val)

# Single input vetor dimension
inp_n = np.shape(x_train)[1]

#model parameters
#learning_rate = 0.001
lrmin = 0.001
lrmax = 0.003
epochs = 2
batch_size = 9000

print('Initial shape of training set: ',np.shape(x_train))

# In[21]:
#reshaping the array to be given as input to 
#for val set only traing set reshaping is done inside the sess object
x_val = x_val[:,np.newaxis,:,np.newaxis]


# In[8]:

x = tf.placeholder(tf.float32, [None,1,inp_n,1])
y_ = tf.placeholder(tf.int32, [None])
y = tf.one_hot(y_, 2)
l = tf.placeholder(tf.float32) #learning rate placeholder


# In[3]:
def lr_fn(i):
    return lrmin + (lrmax-lrmin)*np.exp(-i/epochs)

def conv_(x, wts, bias, stride=1, padding='VALID'):
    x = tf.nn.conv2d(x, wts, [1,1,stride,1], padding)
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)  

def NN_lay(x, wts, bias):
    x = tf.add(tf.matmul(x, wts), bias)
    return tf.nn.relu(x)


# In[4]:

#first conv layer variables
cnw_1 = tf.Variable(tf.truncated_normal([1,3,1,8], mean=0, stddev=0.1)) #stride 1
cnb_1 = tf.Variable(tf.zeros([8]))

#second conv layer variables
cnw_2 = tf.Variable(tf.truncated_normal([1,3,8,16], mean=0, stddev=0.1)) #stride 2
cnb_2 = tf.Variable(tf.zeros([16]))

#third conv layer variables
cnw_3 = tf.Variable(tf.truncated_normal([1,3,16,24], mean=0, stddev=0.1)) #stride 3
cnb_3 = tf.Variable(tf.zeros([24]))

#NN wts and bias
nnwts_1 = tf.Variable(tf.truncated_normal([120, 96], mean=0, stddev=0.1))
nnb_1 = tf.Variable(tf.zeros([96]))

nnwts_2 = tf.Variable(tf.truncated_normal([96, 64], mean=0, stddev=0.1))
nnb_2 = tf.Variable(tf.zeros([64]))

nnwts_3 = tf.Variable(tf.truncated_normal([64, 32], mean=0, stddev=0.1))
nnb_3 = tf.Variable(tf.zeros([32]))

nnwts_4 = tf.Variable(tf.truncated_normal([32, 2], mean=0, stddev=0.1))
nnb_4 = tf.Variable(tf.zeros([2]))


# In[5]:

logits = conv_(x, cnw_1, cnb_1)
#c1 = tf.shape(logits)

logits = conv_(logits, cnw_2, cnb_2, padding='VALID')

logits = tf.nn.max_pool(logits, [1,1,3,1], [1,1,3,1], 'VALID')

logits = conv_(logits, cnw_3, cnb_3,stride=2,padding='VALID')

#padding logits
#logits = tf.pad(logits, tf.convert_to_tensor([[0,0],[0,0],[1,1],[0,0]]))
#after_pad = tf.shape(logits)

logits = tf.nn.max_pool(logits, [1,1,3,1], [1,1,3,1], 'VALID')
#m2 = tf.shape(logits)

# In[6]:
logits = flatten(logits)

logits = NN_lay(logits, nnwts_1, nnb_1)

logits = NN_lay(logits, nnwts_2, nnb_2)

logits = NN_lay(logits, nnwts_3, nnb_3)

logits = tf.add(tf.matmul(logits, nnwts_4), nnb_4)


# In[9]:

#cross entropy loss is objective function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# Optimizer 
optimizer = tf.train.AdamOptimizer(learning_rate=l).minimize(cost)


# In[10]:
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

rec = tf.metrics.recall(tf.argmax(y, 1), tf.argmax(logits,1))
prec = tf.metrics.precision(tf.argmax(y, 1), tf.argmax(logits,1))


# In[11]:

init = tf. global_variables_initializer()
init_l = tf.local_variables_initializer()
#saving the model
save_file = 'New_wts_1/model'
saver = tf.train.Saver()


# In[25]:

with tf.Session() as sess:
    sess.run(init)
    sess.run(init_l)    

    st_time = time.time()
    for epoch in range(epochs):
        strt_tym = time.time()
        
        x_t, y_t = shuffle(x_train, y_train)
        print('------------------------------------------------------')
        print('Epoch: ', epoch)
        us = RandomUnderSampler(ratio=0.7)
        x_t, y_t = us.fit_sample(x_t, y_t)        
        print('Resampled dataset composition {}'.format(Counter(y_t)))        
        
        x_t = x_t[:,np.newaxis,:,np.newaxis]

        no_of_batches = int(len(y_t)/batch_size)
        #print('No of batches: ', no_of_batches)
        
        for offset in range(no_of_batches):
            idx = np.random.randint(0, high=len(y_t), size=batch_size)
            batch_x, batch_y = x_t[idx], y_t[idx]
            
            sess.run(optimizer, feed_dict={x:batch_x, y_:batch_y, l:lr_fn(epoch)})
            loss = sess.run(cost, feed_dict={x:batch_x, y_:batch_y})
        print('Training loss: ', sess.run(cost, feed_dict={x:x_t, y_:y_t}))
        print('Validation Training loss: ', sess.run(cost, feed_dict={x:x_val, y_:y_val}))
        print('  ')
        print('Precision on Validation Set: ', sess.run(prec, feed_dict={x: x_val, y_: y_val}))
        print('Recall on Validation Set: ', sess.run(rec, feed_dict={x: x_val, y_: y_val}))
        print('Validation Accuracy: ', sess.run(accuracy, feed_dict={x:x_val, y_:y_val}))        
        
    saver.save(sess, save_file)
    et_time = time.time()
    print('Total training time (mins): ', (et_time-st_time)/60)
