import pandas as pd
import math
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime
from sklearn import preprocessing
import numpy as np
import joblib
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import yfinance as yf

yf.pdr_override()  # <== that's all it takes :-)

import keras
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History
from sklearn.metrics import mean_squared_error
import pandas_ta as pta

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

N_DAYS_AGO = 1785
train_split = 0.7
data_set_points = 21


def train_test_split_preparation(new_df, data_set_points, train_split):
    new_df = new_df.loc[1:]

    #Preparation of train test set.
    train_indices = int(new_df.shape[0] * train_split)

    train_data = new_df[:train_indices]
    test_data = new_df[train_indices:]
    test_data = test_data.reset_index()
    test_data = test_data.drop(columns=['index'])

    train_arr = train_data.values
    test_arr = test_data.values
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
        [train_data['change'][-(int)(len(y_train) / 10):].copy()])

    y_valid = y_valid.flatten()
    y_valid = np.expand_dims(y_valid, -1)

    X_test = np.array([
        test_arr[i:i + data_set_points]
        for i in range(len(test_arr) - data_set_points)
    ])

    y_test = np.array([
        test_data['change'][i + data_set_points]
        for i in range(len(test_arr) - data_set_points)
    ])

    return X_train, y_train, X_test, y_test, test_data


def lstm_model(X_train, y_train, data_set_points):
    #Setting of seed (to maintain constant result)
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
    adam = optimizers.Adam(learning_rate=0.002)

    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train,
              y=y_train,
              batch_size=15,
              epochs=25,
              shuffle=True,
              validation_split=0.1)

    return model


def save_model(model, path="model.joblib"):
    """Save the model into a .joblib format"""
    joblib.dump(model, path)
    print("model.joblib saved locally", "green")


def get_data(ticker="INFY.BO"):
    today = datetime.now()
    n_days_ago = today - timedelta(days=N_DAYS_AGO)

    start = n_days_ago.date()
    end = today.date()

    # download dataframe
    stock_df = pdr.get_data_yahoo(ticker, start=start, end=end).reset_index()

    stock_df['change'] = (((stock_df['Close'] - stock_df['Close'].shift(1))/stock_df['Close'].shift(1))*100).round(2)
    stock_df=stock_df.dropna()
    stock_df = stock_df[(stock_df['change'] < 4) & (stock_df['change'] > -4)]

    new_df = stock_df.copy()
    new_df = new_df[['change']]
    return new_df



new_df = get_data(ticker="INFY.BO")
#Train test split
X_train, y_train, X_test, y_test, test_data = train_test_split_preparation(new_df, data_set_points, train_split)

#Training of model
model = lstm_model(X_train, y_train, data_set_points)

model.fit(X_train, y_train)
save_model(model, path="Reverse.joblib")

#prediction of model
y_pred = model.predict(X_test)

result = pd.DataFrame(y_test, columns=['true'])
result['pred'] = y_pred
result['error'] = result['true'] - result['pred']


y_true = result["true"]
y_pred = result["pred"]

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

plt.gcf().set_size_inches(30, 15, forward=True)
plt.title(
    'Plot of real price and predicted price against number of days for test set',
    fontname="Times New Roman",
    fontweight="bold",
    fontsize=20)
plt.xlabel('Number of days',
           fontname="Times New Roman",
           fontweight="bold",
           fontsize=20)
plt.ylabel('Adjusted Close Price', fontweight="bold", fontsize=20)

plt.plot(result["true"][-30:], label='Actual Price')
plt.plot(result["pred"][-30:], label='Predicted Price')

#plotting of model
plt.legend(['Actual Price', 'Predicted Price'], prop={'size': 20})
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(result["error"], label='error')
plt.show()


#Sell when model prediction is above 1.5%
df_sell = result[result['pred'] > 1.5]
df_sell['WIN'] = 0
df_sell.loc[df_sell['true'] < 0, 'WIN'] = 1


#Buy when model prediction is below -1.5%
df_buy = result[result['pred'] < -1.5]
df_buy['WIN'] = 0
df_buy.loc[df_buy['true'] > 0, 'WIN'] = 1

plt.plot(df_sell['WIN'])
plt.show()

plt.plot(df_buy['WIN'])
plt.show()
