# -*- coding: utf-8 -*-
"""
Created on Fri May 17 08:25:47 2019

@author: Ferencz Kovacs
"""
import numpy as np
import gzip
import os, sys
import mxnet as mx
mnist = mx.test_utils.get_mnist()
import logging
logging.getLogger().setLevel(logging.INFO)

# Fix the seed
mx.random.seed(43)

# Set the compute context
ctx = mx.cpu()



#Load the data
os.listdir()
os.chdir('E:\\smnist')

with gzip.open('t10k-images-idx3-ubyte.gz', 'r') as f:
            test_data = np.frombuffer(f.read(), np.uint8, offset=16)
            
with gzip.open('t10k-labels-idx1-ubyte.gz', 'r') as f:
            test_labels = np.frombuffer(f.read(), np.uint8, offset = 8)
            
with gzip.open('train-images-idx3-ubyte.gz', 'r') as f:
            train_data = np.frombuffer(f.read(), np.uint8, offset=16)
            
with gzip.open('train-labels-idx1-ubyte.gz', 'r') as f:
            train_labels = np.frombuffer(f.read(), np.uint8, offset = 8)
            
#Reshape and prepare data         
train_data = np.reshape(train_data, (-1,1,10,10))
test_data = np.reshape(test_data, (-1,1,10,10))
batch_size = 100
train_iter = mx.io.NDArrayIter(train_data, train_labels, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(test_data, test_labels, batch_size)


data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(3,3), num_filter=32)
relu1 = mx.sym.Activation(data=conv1, act_type="relu")
pool1 = mx.sym.Pooling(data=relu1, pool_type="max", kernel=(2,2), stride=(2,2))

#second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(3,3), num_filter=64)
relu2 = mx.sym.Activation(data=conv2, act_type="relu")
pool2 = mx.sym.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(2,2))

# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024)
relu3 = mx.sym.Activation(data=fc1, act_type="relu")
dropout = mx.sym.Dropout(data=relu3, p=0.5)
# second fullc
fc2 = mx.sym.FullyConnected(data=dropout, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

lenet_model = mx.mod.Module(symbol=lenet, context=ctx)
# train model
lenet_model.fit(train_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.001},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=50)

test_iter = mx.io.NDArrayIter(test_data, test_labels, batch_size)
# predict accuracy for lenet
acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print(acc)





