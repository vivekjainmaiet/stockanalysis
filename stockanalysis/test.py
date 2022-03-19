from stockanalysis.param import *
import mysql.connector as connection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from pandas_datareader import data as web
import pandas_ta as pta

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import keras
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History


def train_test_split_preparation(new_df, data_set_points, train_split):
    new_df = new_df.loc[1:]

    #Preparation of train test set.
    train_indices = int(new_df.shape[0] * train_split)

    train_data = new_df[:train_indices]
    test_data = new_df[train_indices:]
    test_data = test_data.reset_index()
    test_data = test_data.drop(columns=['index'])

    train_arr = np.diff(train_data.loc[:, :].values, axis=0)
    test_arr = np.diff(test_data.loc[:, :].values, axis=0)

    X_train = np.array([
        train_arr[i:i + data_set_points]
        for i in range(len(train_arr) - data_set_points)
    ])

    y_train = np.array([
        train_arr[i + data_set_points]
        for i in range(len(train_arr) - data_set_points)
    ])

    y_valid = np.array(
        [train_data['Close'][-(int)(len(y_train) / 10):].copy()])

    y_valid = y_valid.flatten()
    y_valid = np.expand_dims(y_valid, -1)

    X_test = np.array([
        test_arr[i:i + data_set_points]
        for i in range(len(test_arr) - data_set_points)
    ])

    y_test = np.array([
        test_data['Close'][i + data_set_points]
        for i in range(len(test_arr) - data_set_points)
    ])

    return X_train, y_train, X_test, y_test, test_data


def lstm_model(X_train, y_train, data_set_points):
    tf.random.set_seed(20)
    np.random.seed(10)

    lstm_input = Input(shape=(data_set_points, X_train.shape[2]),
                       name='input_for_lstm')

    inputs = LSTM(21, name='first_layer', return_sequences=True)(lstm_input)

    inputs = Dropout(0.1, name='first_dropout_layer')(inputs)
    inputs = LSTM(32, name='lstm_1')(inputs)
    inputs = Dropout(0.05, name='lstm_dropout_1')(
        inputs)  #Dropout layers to prevent overfitting
    inputs = Dense(32, name='first_dense_layer')(inputs)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.002)

    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train,
              y=y_train,
              batch_size=15,
              epochs=25,
              shuffle=True,
              validation_split=0.1)

    return model


conn = connection.connect(**config)
mycursor = conn.cursor(dictionary=True)

query = f"SELECT * FROM stocksdb.StocksList;"
mycursor.execute(query)
stock_list = mycursor.fetchall()


for stock in stock_list:
    stock_id = stock['ID']
    query = f"SELECT * FROM stocksdb.raw_technical WHERE stock_id={stock_id};"
    mycursor.execute(query)
    stock_technicals = mycursor.fetchall()
    df = pd.DataFrame(stock_technicals)[COLUMNS]
    train_split = 0.7
    data_set_points = 21
    X_train, y_train, X_test, y_test, test_data = train_test_split_preparation(df, data_set_points, train_split)

    #Training of model
    model = lstm_model(X_train, y_train, data_set_points)

    #prediction of model
    y_pred = model.predict(X_test)
    re  = pd.DataFrame(y_test,columns=['true'])
    re['pred'] = y_pred
    re['yesterday'] = re['true'].shift(1)
    re = re[1:]
    re['true_pred'] = re['yesterday'] + re['pred']
    re['error'] = re['true_pred'] - re['true']
    import math

    y_true = re["true"]
    y_pred = re["true_pred"]

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    rsquared = r2_score(y_true, y_pred)
    max_err = max_error(y_true, y_pred)

    print('MSE =', round(mse, 2))
    print('RMSE =', round(rmse, 2))
    print('MAE =', round(mae, 2))
    print('R2 =', round(rsquared, 2))
    print('Max Error =', round(max_err, 2))

    plt.plot(re["true"], label='Actual Price')
    plt.plot(re["true_pred"], label='Predicted Price')
    plt.show()
    plt.plot(re["error"], label='Residual')
    plt.show()

    print(re)

    break
