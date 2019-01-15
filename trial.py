# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:45:27 2018

@author: kenn
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import functools
import math

#load the data
def read_csv(path):
    return pd.read_csv(path, index_col=[0],
                       dtype={'Open': np.float64, 'High': np.float64, 'Low': np.float64,
                              'Close': np.float64, 'Adj Close': np.float64,'Volume': np.float64}
                      )
data_aus_aord = read_csv('data/^AORD.csv')
data_jap_n225 = read_csv('data/^N225.csv')
data_hk_hs = read_csv('data/^HSI.csv')
data_ger_dax = read_csv('data/^GDAXI.csv')
data_us_nyse = read_csv('data/^NYA.csv')
data_us_dji = read_csv('data/^DJI.csv')
data_us_sap = read_csv('data/^GSPC.csv')

#cut only the interesting features
closing_data = pd.DataFrame()

closing_data['aord_close'] = data_aus_aord['Adj Close']
closing_data['n225_close'] = data_jap_n225['Adj Close']
closing_data['hs_close']   = data_hk_hs['Adj Close']
closing_data['dax_close']  = data_ger_dax['Adj Close']
closing_data['nyse_close'] = data_us_nyse['Adj Close']
closing_data['dji_close']  = data_us_dji['Adj Close']
closing_data['sap_close']  = data_us_sap['Adj Close']

#cleaning the data
# b) Fill gaps in the data.
closing_data = closing_data.fillna(method='ffill')
closing_data = closing_data.fillna(0)

# observe data
print(closing_data.describe())
# observe data in graph
_ = closing_data.plot(figsize=(20,10))

# => data needs to be modifiend to fit the same scale

# c) Normalize data to fit the same scale
closing_data_norm = pd.DataFrame()

closing_data_norm['aord_close'] = closing_data['aord_close'] / max(closing_data['aord_close'])
closing_data_norm['n225_close'] = closing_data['n225_close'] / max(closing_data['n225_close'])
closing_data_norm['hs_close']   = closing_data['hs_close']   / max(closing_data['hs_close'])
closing_data_norm['dax_close']  = closing_data['dax_close']  / max(closing_data['dax_close'])
closing_data_norm['nyse_close'] = closing_data['nyse_close'] / max(closing_data['nyse_close'])
closing_data_norm['dji_close']  = closing_data['dji_close']  / max(closing_data['dji_close'])
closing_data_norm['sap_close']  = closing_data['sap_close']  / max(closing_data['sap_close'])

# observe trend in data in graph
_ = closing_data_norm.plot(figsize=(20,10))

# observe corelations in data in graph
fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(15)

_ = pd.plotting.autocorrelation_plot(closing_data_norm['aord_close'], label='aord_close')
_ = pd.plotting.autocorrelation_plot(closing_data_norm['n225_close'], label='n225_close')
_ = pd.plotting.autocorrelation_plot(closing_data_norm['hs_close'], label='hs_close')
_ = pd.plotting.autocorrelation_plot(closing_data_norm['dax_close'], label='dax_close')
_ = pd.plotting.autocorrelation_plot(closing_data_norm['nyse_close'], label='nyse_close')
_ = pd.plotting.autocorrelation_plot(closing_data_norm['dji_close'], label='dji_close')
_ = pd.plotting.autocorrelation_plot(closing_data_norm['sap_close'], label='sap_close')

_ = plt.legend(loc='best')

# => strong corelations in smaller lag (up to ~400 lag), this meand resent past data corelates with upcomming data
# observe corelations in data in matrix graph
_ = pd.plotting.scatter_matrix(closing_data_norm, figsize=(10, 10), diagonal='kde')

# => we can see there is significant corellation between market indexes on the same day
# make time series data stationary in the mean, thus having no trend in the data
# aplying log(Vt/Vt-1) on all data
closing_data_norm_log = pd.DataFrame()

closing_data_norm_log['aord_close'] = np.log(closing_data['aord_close'] / closing_data['aord_close'].shift())
closing_data_norm_log['n225_close'] = np.log(closing_data['n225_close'] / closing_data['n225_close'].shift())
closing_data_norm_log['hs_close'] = np.log(closing_data['hs_close'] / closing_data['hs_close'].shift())
closing_data_norm_log['dax_close'] = np.log(closing_data['dax_close'] / closing_data['dax_close'].shift())
closing_data_norm_log['nyse_close'] = np.log(closing_data['nyse_close'] / closing_data['nyse_close'].shift())
closing_data_norm_log['dji_close'] = np.log(closing_data['dji_close'] / closing_data['dji_close'].shift())
closing_data_norm_log['sap_close'] = np.log(closing_data['sap_close'] / closing_data['sap_close'].shift())

# remove first row (contains NaN because of the t-1 shift)
closing_data_norm_log = closing_data_norm_log.iloc[1:]
print(closing_data_norm_log.describe())

#creating trianing/testing dataset
closing_data_norm_log['sap_rising'] = 0
closing_data_norm_log['sap_falling'] = 0

closing_data_norm_log.loc[closing_data_norm_log['sap_close'] >= 0, 'sap_rising'] = 1
closing_data_norm_log.loc[closing_data_norm_log['sap_close'] < 0, 'sap_falling'] = 1

print(closing_data_norm_log.describe())

#picking features and designing a dataframe
# including labels one-hot encoded (sap_rising, sap_falling)
feature_columns = ['sap_rising', 'sap_falling',
                  'aord_close_t0', 'aord_close_t1', 'aord_close_t2',
                  'n225_close_t0', 'n225_close_t1', 'n225_close_t2',
                  'hs_close_t0', 'hs_close_t1', 'hs_close_t2',
                  'dax_close_t0', 'dax_close_t1', 'dax_close_t2',
                  'nyse_close_t1', 'nyse_close_t2', 'nyse_close_t3',
                  'dji_close_t1', 'dji_close_t2', 'dji_close_t3',
                  'sap_close_t1', 'sap_close_t2', 'sap_close_t3']
dataset = pd.DataFrame(columns=feature_columns)

# compose dataset from features
for i in range(3, len(closing_data_norm_log)):
    dataset = dataset.append({
        'sap_rising':    closing_data_norm_log.iloc[i]['sap_rising'],
        'sap_falling':   closing_data_norm_log.iloc[i]['sap_falling'],
        'aord_close_t0': closing_data_norm_log.iloc[i]['aord_close'],
        'aord_close_t1': closing_data_norm_log.iloc[i-1]['aord_close'],
        'aord_close_t2': closing_data_norm_log.iloc[i-2]['aord_close'],
        'n225_close_t0': closing_data_norm_log.iloc[i]['n225_close'],
        'n225_close_t1': closing_data_norm_log.iloc[i-1]['n225_close'],
        'n225_close_t2': closing_data_norm_log.iloc[i-2]['n225_close'],
        'hs_close_t0':   closing_data_norm_log.iloc[i]['hs_close'],
        'hs_close_t1':   closing_data_norm_log.iloc[i-1]['hs_close'],
        'hs_close_t2':   closing_data_norm_log.iloc[i-2]['hs_close'],
        'dax_close_t0':  closing_data_norm_log.iloc[i]['dax_close'],
        'dax_close_t1':  closing_data_norm_log.iloc[i-1]['dax_close'],
        'dax_close_t2':  closing_data_norm_log.iloc[i-2]['dax_close'],
        'nyse_close_t1': closing_data_norm_log.iloc[i-1]['nyse_close'],
        'nyse_close_t2': closing_data_norm_log.iloc[i-2]['nyse_close'],
        'nyse_close_t3': closing_data_norm_log.iloc[i-3]['nyse_close'],
        'dji_close_t1':  closing_data_norm_log.iloc[i-1]['dji_close'],
        'dji_close_t2':  closing_data_norm_log.iloc[i-2]['dji_close'],
        'dji_close_t3':  closing_data_norm_log.iloc[i-3]['dji_close'],
        'sap_close_t1':  closing_data_norm_log.iloc[i-1]['sap_close'],
        'sap_close_t2':  closing_data_norm_log.iloc[i-2]['sap_close'],
        'sap_close_t3':  closing_data_norm_log.iloc[i-3]['sap_close']},
        ignore_index=True
    )
    print(dataset.describe())
    
class DataProvider():
    def __init__(self, dataset, batch_size):
        self.ctr = 0
        self.batch_size = batch_size
        
        # split training/testing according to ratio (default 0.8)
        train_set_size = int(len(dataset) * 0.8)
        test_set_size = len(dataset) - train_set_size

        self.training_dataset = dataset[:train_set_size]
        self.testing_dataset  = dataset[train_set_size:]

        # split labels
        self.training_labels = self.training_dataset[self.training_dataset.columns[:2]]
        self.training_dataset = self.training_dataset[self.training_dataset.columns[2:]]
        self.testing_labels = self.testing_dataset[self.testing_dataset.columns[:2]]
        self.testing_dataset = self.testing_dataset[self.testing_dataset.columns[2:]]
        
    def next_batch_train(self):
        begin_position = self.ctr * self.batch_size
        
        if begin_position + self.batch_size >= len(self.training_dataset):
            data = self.training_dataset[begin_position:]
            label = self.training_labels[begin_position:]
            self.ctr = 0
        else:
            data = self.training_dataset[begin_position:begin_position + self.batch_size]
            label = self.training_labels[begin_position:begin_position + self.batch_size]
            self.ctr += 1
        
        return data.values, label.values
    
    def get_test_data(self):
        return self.testing_dataset.values, self.testing_labels.values

#Creating the model
        # config
no_of_iterations = 50000
batch_size = 200

## model config
hidden_layer1_neurons = 60
hidden_layer2_neurons = 30
hidden_layer3_neurons = 20

# DropOut
pkeep_train = 0.75

# number of features
input_dim = len(dataset.columns) - 2

# number of output classes
output_dim = 2
data_provider = DataProvider(dataset, batch_size)

# custom decorator for Model
#  - to make functions execute only the first time (every time the functions are called, the graph would be extended by new code)
#  - name the variable scope for TF visualization
def define_scope(function, scope=None):
    attribute = '_cache_' + function.__name__

    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class Model():
    def __init__(self, data, label, learning_rate):
        self.data = data
        self.label = label
        self.learning_rate = learning_rate
        self.prediction
        self.optimize
        
    @define_scope
    def prediction(self):        
        # weights + biases
        w1 = tf.Variable(tf.truncated_normal([input_dim, hidden_layer1_neurons], stddev=0.0001))
        b1 = tf.Variable(tf.ones([hidden_layer1_neurons]))

        w2 = tf.Variable(tf.truncated_normal([hidden_layer1_neurons, hidden_layer2_neurons], stddev=0.0001))
        b2 = tf.Variable(tf.ones([hidden_layer2_neurons]))

        w3 = tf.Variable(tf.truncated_normal([hidden_layer2_neurons, hidden_layer3_neurons], stddev=0.0001))
        b3 = tf.Variable(tf.ones([hidden_layer3_neurons]))
        
        w4 = tf.Variable(tf.truncated_normal([hidden_layer3_neurons, output_dim], stddev=0.0001))
        b4 = tf.Variable(tf.ones([output_dim]))
        
        # hidden layers
        Y1 = tf.nn.relu(tf.matmul(self.data, w1) + b1)
        Y1d = tf.nn.dropout(Y1, pkeep)
        Y2 = tf.nn.relu(tf.matmul(Y1, w2) + b2)
        Y2d = tf.nn.dropout(Y2, pkeep)
        Y3 = tf.nn.relu(tf.matmul(Y2, w3) + b3)
        Y3d = tf.nn.dropout(Y3, pkeep)
        
        # softmax layer
        return tf.nn.softmax(tf.matmul(Y3d, w4) + b4)
    
    @define_scope
    def optimize(self):
        # compute cost function and minimize
        cross_entropy = -tf.reduce_sum(self.label * tf.log(self.prediction))
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy), cross_entropy
    
    @define_scope
    def error(self):
        mistakes = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        loss = -tf.reduce_sum(self.label * tf.log(self.prediction))
        return accuracy, loss
    
    # data feed
X = tf.placeholder(tf.float32, [None, input_dim])
_Y = tf.placeholder(tf.float32, [None, output_dim])
learning_rate = tf.placeholder(tf.float32)

# DropOut: feed in 1 when testing, 0.75 when training
pkeep = tf.placeholder(tf.float32)

model = Model(data=X, label=_Y, learning_rate=learning_rate)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

accuracy = []
_loss = []
for i in range(no_of_iterations):
    # execute training step
    # optimizer learning rate decay
    lrmax = 0.001
    lrmin = 0.00001
    lr = lrmin + (lrmax - lrmin) * math.exp(-i / 2000)
    
    data_batch, label_batch = data_provider.next_batch_train()
    sess.run(model.optimize, feed_dict={X: data_batch, _Y: label_batch, learning_rate:lr, pkeep: pkeep_train})
    
    if i % 500 == 0:
        # compute accuracy
        data_batch, label_batch = data_provider.get_test_data()
        acc, loss = sess.run(model.error, feed_dict={X: data_batch, _Y: label_batch, pkeep: 1})
        accuracy.append(acc)
        _loss.append(loss)
        print('---epoch {}---\naccuracy: {}, loss: {}'.format(i // 500, acc, loss))

print('Training finished')

# accuracy on test data
data_batch, label_batch = data_provider.get_test_data()
acc, loss = sess.run(model.error, feed_dict={X: data_batch, _Y: label_batch, pkeep: 1})
print('Test: accuracy={}, loss={}'.format(acc, loss))

plt.figure(figsize=(3,6))

# accuracy
plt.subplot(211)
plt.plot(accuracy)

# loss
plt.subplot(212)
plt.plot(np.log(_loss))

plt.show()
