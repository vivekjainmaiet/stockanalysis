import re
import numpy as np
import pandas_ta as pta
import pandas as pd
import yfinance as yf
import pytz
import joblib

from datetime import datetime# Get the stock quote
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import set_config; set_config(display='diagram')
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from stockanalysis.data import *
from stockanalysis.encoder import *


class Trainer:

    def __init__(self, ticker):
        self.ticker = ticker


if __name__ == "__main__":
    trainer = Trainer('TCS', pages=2)
    #Get the data
    data = get_technical(symbol="INFY.NS", period='5y')
    #Clean the data
    cleaned_data = clean_data(data)
    #Scale the data
    scaled_data, pipe = set_pipeline(cleaned_data)
    #Split data in trainning and testing
    x_train, y_train, x_test, y_test = split_timeseries(scaled_data, data)
    #Create Model
    model = create_model(x_train)
    #Train Model
    model = train(x_train, y_train, model)
    #Evaluate Model
    mpe = evaluate(x_test, y_test, model)
    #Print Root Mean Square Error
    print(mpe)
    #Save Model
    save_model(model)
