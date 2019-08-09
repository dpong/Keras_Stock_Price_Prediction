from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import os, pickle
from matplotlib import style

class Timeseries_tf():
    def __init__(self):
        self.df = pd.DataFrame()

    def get_data(self):
        self.df = pdr.DataReader('^TWII','yahoo')
        self.df.drop(columns=['Adj Close'],inplace=True)
        #print(self.df.columns)
        #['High', 'Low', 'Open', 'Close', 'Volume']
    def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, single_step=False):
        #整理資料
        data = []
        labels = []
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            data.append(dataset[indices])
            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])
        return np.array(data), np.array(labels)

    def handle_data(self):
        
        TRAIN_SPLIT = int(len(self.df) * 0.8)   #80%來train
        tf.random.set_seed(13)
        self.df['HL_PCT'] = (self.df['High']-self.df['Low'])/self.df['Close'] * 100
        self.df['PCT_change'] = (self.df['Close']-self.df['Open'])/self.df['Open'] * 100
        features = self.df
        self.close_mean = features['Close'].mean()    #標準化們
        self.close_std = features['Close'].std()
        dataset = features.values
        dataset_mean = dataset.mean(axis=0)
        dataset_std = dataset.std(axis=0)
        dataset = (dataset -dataset_mean) / dataset_std
        self.past_history = 60      #訓練用的過去天數
        self.future_target = 10     #預測未來天數
        #target是Close, 就是dataset的第4個
        self.x_train, self.y_train = Timeseries_tf.multivariate_data(dataset, dataset[:,3], 0,
                                                    TRAIN_SPLIT, self.past_history,
                                                    self.future_target)
        self.x_val, self.y_val = Timeseries_tf.multivariate_data(dataset, dataset[:,3],
                                                    TRAIN_SPLIT, None, self.past_history,
                                                    self.future_target)
        BATCH_SIZE = 256
        BUFFER_SIZE = 10000
        train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        val_data = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.val_data = val_data.batch(BATCH_SIZE).repeat()
    
    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(32,return_sequences=True,input_shape=self.x_train.shape[-2:]))
        self.model.add(tf.keras.layers.LSTM(16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.future_target))
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        #測試model的輸出shape
        #for x, y in self.val_data.take(1): 
        #    print(self.model.predict(x).shape)
    
    def training(self):
        EVALUATION_INTERVAL = 100
        EPOCHS = 20
        self.model.fit(self.train_data, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=self.val_data, validation_steps=50)
        self.model.save('timeseries_model.h5')

    def evaluation(self):
        self.model = keras.models.load_model('timeseries_model.h5')
        results = self.model.evaluate(self.x_val_uni,self.y_val_uni)
        print('test loss, test acc:', results)

    def prediction_test(self):
        self.model = keras.models.load_model('timeseries_model.h5')
        for x, y in self.val_data.take(3):
            predict = self.model.predict(x) * self.close_std + self.close_mean
            x_p = x[0][:,3] * self.close_std + self.close_mean
            y_p = y[0] * self.close_std + self.close_mean
            plot = v.show_plot(x_p, y_p, predict[0])

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
        plt.title('LSTM Model')
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
    #t.build_model()
    #t.training()
    #t.evaluation()
    t.prediction_test()



