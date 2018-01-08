import pickle
import time
import numpy as np
import tensorflow as tf

from collections import Counter

from sklearn.utils import shuffle

from imblearn.under_sampling import RandomUnderSampler 

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
lrmin = 0.005
lrmax = 0.005
epochs = 50
batch_size = 9000
rat = 1.5

frac_inc = 4

# In[8]:

x = tf.placeholder(tf.float32, [None,inp_n])
y_ = tf.placeholder(tf.int32, [None])
y = tf.one_hot(y_, 2)
l = tf.placeholder(tf.float32) #learning rate placeholder

# In[3]:
def lr_fn(i):
    return lrmin + (lrmax-lrmin)*np.exp(-i/epochs)

def NN_lay(x, wts, bias):
    x = tf.add(tf.matmul(x, wts), bias)
    x = tf.nn.relu(x)
    return tf.nn.dropout(x, keep_prob=0.9)


#NN wts and bias
nnwts_1 = tf.Variable(tf.truncated_normal([inp_n, frac_inc*inp_n], mean=0, stddev=0.1))
nnb_1 = tf.Variable(tf.zeros([frac_inc*inp_n]))

nnwts_2 = tf.Variable(tf.truncated_normal([frac_inc*inp_n, 64], mean=0, stddev=0.1))
nnb_2 = tf.Variable(tf.zeros([64]))

nnwts_3 = tf.Variable(tf.truncated_normal([64, 32], mean=0, stddev=0.1))
nnb_3 = tf.Variable(tf.zeros([32]))

nnwts_4 = tf.Variable(tf.truncated_normal([32, 2], mean=0, stddev=0.1))
nnb_4 = tf.Variable(tf.zeros([2]))


logits = NN_lay(x, nnwts_1, nnb_1)

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
save_file = 'saved_us/model'
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

        
        us = RandomUnderSampler(ratio={0:int(rat*np.sum(y_t)), 1:np.sum(y_t)})
        x_t, y_t = us.fit_sample(x_t, y_t)        
        print('Resampled dataset composition {}'.format(Counter(y_t)))        
        
        no_of_batches = int(len(y_t)/batch_size)
        #print('No of batches: ', no_of_batches)
        
        for offset in range(no_of_batches):
            idx = np.random.randint(0, high=len(y_t), size=batch_size)
            batch_x, batch_y = x_t[idx], y_t[idx]
            
            sess.run(optimizer, feed_dict={x:batch_x, y_:batch_y, l:lr_fn(epoch)})
            
        #print('Training loss: ', sess.run(cost, feed_dict={x:x_t, y_:y_t}))
        #print('Validation Training loss: ', sess.run(cost, feed_dict={x:x_val, y_:y_val}))
        #print('  ')
        print('Precision on Validation Set: ', sess.run(prec, feed_dict={x: x_val, y_: y_val}))
        print('Recall on Validation Set: ', sess.run(rec, feed_dict={x: x_val, y_: y_val}))
        print('Validation Accuracy: ', sess.run(accuracy, feed_dict={x:x_val, y_:y_val}))        
        
    saver.save(sess, save_file)
    et_time = time.time()
    print('Total training time (mins): ', (et_time-st_time)/60)
