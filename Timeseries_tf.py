from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, GRU, Dropout
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.keras.utils import Progbar
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import os, pickle
from matplotlib import style
from data import *


config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
tf.compat.v1.Session(config=config)

class Timeseries_tf():
    def __init__(self):
        self.df = pd.DataFrame()
        self.past_history = 90      #訓練用的過去天數
        self.future_target = 30     #預測未來天數
        self.col = 7
        self.checkpoint_path = 'model_weights/weights'
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.check_index = self.checkpoint_path + '.index'
        self.model = self.build_model()
        self.epochs = 500
        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.optimizer = tf.optimizers.Adam(learning_rate=0.0001, epsilon=0.000065)
        self.loss_function = tf.keras.losses.MSE
        self.bar = Progbar(self.epochs)

    def get_data(self):
        self.df = get_crypto_from_api('BTC', 2000, 'day')
        self.df['HL_PCT'] = (self.df['High']-self.df['Low'])/self.df['Close'] * 100
        self.df['PCT_change'] = (self.df['Close']-self.df['Open'])/self.df['Open'] * 100
        
    def _multivariate_data(self, dataset, target, start_index, end_index, single_step=False):
        #整理資料
        data = []
        labels = []
        start_index = start_index + self.past_history
        if end_index is None:
            end_index = len(dataset) - self.future_target
        for i in range(start_index, end_index):
            indices = range(i-self.past_history, i)
            data.append(dataset[indices])
            if single_step:
                labels.append(target[i+self.future_target])
            else:
                labels.append(target[i:i+self.future_target])
        return np.array(data), np.array(labels)

    def handle_data(self):
        TRAIN_SPLIT = int(len(self.df) * 0.8)   #80%來train
        features = self.df
        self.close_mean = features['Close'].mean()    #標準化們
        self.close_std = features['Close'].std()
        dataset = features.values
        dataset_mean = dataset.mean(axis=0)
        dataset_std = dataset.std(axis=0)
        dataset = (dataset -dataset_mean) / dataset_std
        #target是Close, 就是dataset的第0個
        x_train, y_train = self._multivariate_data(dataset, dataset[:,0], 0, TRAIN_SPLIT)
        x_val, y_val = self._multivariate_data(dataset, dataset[:,0], TRAIN_SPLIT, None)
        self.train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)
        self.val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(10000).batch(128)


    def build_GRU_model(self):
        tf.keras.backend.set_floatx('float64')
        data_input = Input(shape=(self.past_history, self.col), name='input')
        gru1 = GRU(64, return_sequences=True)(data_input)
        drop1 = Dropout(0.2)(gru1)
        gru2 = GRU(64, return_sequences=False)(drop1)
        drop2 = Dropout(0.2)(gru2)
        #gru3 = GRU(64, return_sequences=True)(drop2)
        #drop3 = Dropout(0.2)(gru3)
        #gru4 = GRU(64, return_sequences=False)(drop3)
        #drop4 = Dropout(0.2)(gru4)
        d1 = Dense(self.future_target, activation='relu')(drop2)
        model = Model(inputs=data_input, outputs=d1)
        if os.path.exists(self.check_index):
            print('-'*52+'  Weights loaded!!'+'-'*52)
            model.load_weights(self.checkpoint_path)
        else:
            print('-'*53+'Create new model!!'+'-'*53)
        return model


    def build_model(self):
        tf.keras.backend.set_floatx('float64')
        data_input = Input(shape=(self.past_history, self.col), name='input')
        con1 = Conv1D(64 , 10, padding='causal')(data_input)
        con1_norm = BatchNormalization()(con1)
        con1_norm_act = Activation('elu')(con1_norm)
        con2 = Conv1D(64 , 10, padding='causal')(con1_norm_act)
        con2_norm = BatchNormalization()(con2)
        con2_norm_act = Activation('elu')(con2_norm)
        pool_max = MaxPooling1D(pool_size=5, strides=1, padding='same')(con2_norm_act)
        con3 = Conv1D(64 , 10, padding='causal')(pool_max)
        con3_norm = BatchNormalization()(con3)
        con3_norm_act = Activation('elu')(con3_norm)
        pool_max_2 = MaxPooling1D(pool_size=5, strides=1, padding='same')(con3_norm_act)
        flat = Flatten()(pool_max_2)
        d1 = Dense(self.future_target, activation='relu')(flat)
        model = Model(inputs=data_input, outputs=d1)
        if os.path.exists(self.check_index):
            print('-'*52+'  Weights loaded!!'+'-'*52)
            model.load_weights(self.checkpoint_path)
        else:
            print('-'*53+'Create new model!!'+'-'*53)
        return model
    
    # loss
    def _loss(self, model, x, y):
        y_ = self.model(x)
        return self.loss_function(y_true=y, y_pred=y_)

    # gradient
    def _grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self._loss(self.model, inputs, targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def training(self):
        for i in range(self.epochs):
            for x, y in self.train_data:
                loss_value, grads = self._grad(self.model, x, y)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables),get_or_create_global_step())
                self.epoch_loss_avg(loss_value)
            self.bar.update(i, values=[('loss', self.epoch_loss_avg.result().numpy())])
        self.model.save_weights(self.checkpoint_path, save_format='tf')

    def prediction_test(self):
        print(self.val_data)
        for x, y in self.val_data.take(10):
            raw_predict = self.model(x)
            predict = raw_predict.numpy() # * self.close_std + self.close_mean
            x = x.numpy()
            y = y.numpy()
            x_p = x[1][:,0] #* self.close_std + self.close_mean
            y_p = y[1] #* self.close_std + self.close_mean
            plot = v.show_plot(x_p, y_p, predict[1])
        

class Visualize():
    def __init__(self):
        mpl.rcParams['figure.figsize'] = (8, 6)
        mpl.rcParams['axes.grid'] = False
        style.use('ggplot')

    def show_plot(self, history, true_future, prediction):
        plt.figure(figsize=(12, 6))            
        num_in = Visualize.create_time_steps(len(history))
        num_out = len(true_future)
        plt.plot(num_in, np.array(history),c='b', label='History')
        plt.plot(np.arange(num_out), np.array(true_future), c='g',ls='--', label='True Future')
        if prediction.any():
            plt.plot(np.arange(num_out), np.array(prediction), c='r',ls='-.', label='Predicted Future')
        plt.title('Model')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        plt.show()

    def create_time_steps(length):
        time_steps = []
        for i in range(-length, 0, 1):
          time_steps.append(i)
        return time_steps

if __name__=='__main__':
    t = Timeseries_tf()
    v = Visualize()
    t.get_data()
    t.handle_data()
    t.training()
    #t.prediction_test()



