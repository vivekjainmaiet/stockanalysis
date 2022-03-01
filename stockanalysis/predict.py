import os
from math import sqrt
import numpy as np

import joblib
import pandas as pd
from TaxiFareModel.params import MODEL_NAME
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
from stockanalysis.data import get_technical
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector


PATH_TO_LOCAL_MODEL = 'model.joblib'
BUCKET_NAME = "one-stop-stock-analysis"

def split_predict(scaled_data, data):
    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(data[i])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_train.shape

    #print(y_train.shape)
    return x_train,y_train

def get_test_data(ticker="INFY.NS"):
    start_date = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    df = get_technical(symbol=ticker, start=start_date, end=end_date)
    return df


def set_pipeline(cleaned_data):
    '''returns a pipelined model'''
    cleaned_data = cleaned_data[['Open', 'High', 'Low', 'Close', 'Volume',
                       'ema12', 'ema21', 'ema26', 'ema34', 'ema55', 'ema99',
                       'ema200', 'hma12', 'hma21', 'hma26', 'hma34', 'hma55',
                       'hma99', 'hma200', 'rsi', 'atr', 'bb_upper', 'bb_lower',
                       'macd_signal', 'macd_line', 'adx', 'vwap']]
    scale = StandardScaler()
    scaled_data = scale.fit_transform(cleaned_data)
    return scaled_data, scale


def download_model(storage_location='models/stockanalysis/Pipeline/INFY.NS.joblib',
        bucket=BUCKET_NAME,
        rm=True):
    client = storage.Client().bucket(bucket)
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model


def get_model(path_to_joblib="/Users/vivek/code/vivekjainmaiet/stockanalysis/model.joblib"):
    model = joblib.load('model.joblib')
    return model


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


if __name__ == '__main__':
    #model = download_model(storage_location='models/stockanalysis/Pipeline/INFY.NS.joblib',bucket=BUCKET_NAME,m=True)
    model = get_model()
    df = get_test_data()
    cleaned_data = df.drop(columns=['Date']).tail(60)
    scale = StandardScaler()
    scaled_data = scale.fit_transform(cleaned_data)
    print(scaled_data)
    #x_test, y_test = split_predict(scaled_data,cleaned_data['Close'].to_numpy())
    #X = cleaned_data.to_numpy()[-60:, :]
    #breakpoint()
    #X, y = split_predict(scaled_data, X)
    #results = model.predict(X)
    #pred = float(results)

    #print(scale.inverse_transform(model.predict(scaled_data)))
