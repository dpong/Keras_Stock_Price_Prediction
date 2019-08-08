#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:29:45 2019

@author: dpong
"""

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM, Dropout
from keras import metrics
from datetime import datetime, date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import style


class Keras_model():
    def __init__ (self,timestep=60):   #過去60天來訓練未來幾天的預測
        self.df = pd.DataFrame()
        self.df_origin = pd.DataFrame()
        self.train_accuracy = None
        self.predict_date_start = None
        self.predict_length = None
        self.predict_df = pd.DataFrame()
        self.scale = MinMaxScaler()
        self.model = Sequential()
        self.timestep = timestep  #LSTM的timestep參數
        
    def get_data(self,ticker,forecast_out=1):
        self.predict_length = forecast_out #提供查詢
        self.ticker = ticker
        self.df_origin = pdr.DataReader(self.ticker,'yahoo')
        #用talib一定要rename一下key才會對應的到
        self.df_origin.rename(columns = {'Open':'open',
                                         'High':'high',
                                         'Low':'low',
                                         'Close':'close'}, inplace = True)
        self.df_origin.drop(columns=['Adj Close'],inplace=True)
        self.df_origin['HL_PCT'] = (self.df_origin['high']-self.df_origin['low'])/self.df_origin['close'] * 100
        self.df_origin['PCT_change'] = (self.df_origin['close']-self.df_origin['open'])/self.df_origin['open'] * 100
        #self.df_origin['5_ma'] = self.df_origin['close'].rolling(window=5).mean()
        self.df = self.df_origin[:-self.predict_length].copy()    #避免pandas一直跳警告
        
    def training(self):
        
        self.predict_date_start = datetime.strptime(str(self.df.iloc[-self.predict_length].name), "%Y-%m-%d %H:%M:%S")
        self.predict_date_start = datetime.date(self.predict_date_start)
        test_df = self.df.dropna(how='any').copy()
        test_df['label'] = self.df['close'].shift(-self.predict_length)
        X = np.array(test_df.drop(columns=['label']))    
        X = X[:-self.predict_length] 
        X = self.scale.fit_transform(X)  #NN都要scale一下
        y = np.array(test_df['label'].dropna(how='any'))  
        X_train = []   #預測點的前 60 天的資料
        y_train = []   #預測點
        for i in range(self.timestep, len(X)):  
            X_train.append(X[i-self.timestep:i])
            y_train.append(y[i-self.predict_length:i])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
        
        self.model.add(LSTM(50, return_sequences = True,input_shape = (X_train.shape[1], X_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 50))
        self.model.add(Dropout(0.2))
        #self.model.add(LSTM(units = 40))
        #self.model.add(Dropout(0.2))
        #self.model.add(Dense(30, activation='relu'))
        self.model.add(Dense(self.predict_length))
        self.model.compile(optimizer ='adam', loss = 'mean_squared_error', 
                      metrics =[metrics.mae])
        self.model.summary()
        #訓練
        hist = self.model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=10, batch_size=32)
        #存檔
        self.model.save('{}_train_model.h5'.format(self.ticker))
        
    def prediction_test(self,show_days=20):
        
        self.model = load_model('{}_train_model.h5'.format(self.ticker))
        
        self.predict_df = self.df[-(self.predict_length+self.timestep):].copy()
        X = np.array(self.predict_df[:-self.predict_length])
        X = self.scale.fit_transform(X)  #NN都要scale一下
        X_predict = []
        for i in range(self.timestep, self.timestep+30): #30個sample   
            X_predict.append(X)
        X_predict = np.array(X_predict)
        y = self.model.predict(X_predict)
        #資料整理
        pre_y = np.full(self.predict_length+self.timestep,np.nan)
        y = np.append(pre_y,y)
        self.predict_df['prediction'] = y
        self.df['prediction'] = np.nan
        self.df = self.df[:-self.predict_length-1]
        self.df = pd.concat([self.df,self.predict_df],axis=0)
        #畫圖
        style.use('ggplot')
        self.df['close'][-show_days:].plot()
        self.df['prediction'][-show_days:].plot()
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('{}'.format(self.ticker))
                

if __name__=='__main__':
    k = Keras_model()
    k.get_data('^TWII',forecast_out=1)
    k.training()
    k.prediction_test()
    