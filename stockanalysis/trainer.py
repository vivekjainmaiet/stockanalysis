import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta

import pandas_ta as pta


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

import joblib
from google.cloud import storage

from stockanalysis.data import *
from stockanalysis.encoder import *
from stockanalysis.utils import *
from stockanalysis.param import *


class Trainer():

    def __init__(self, ticker):
        self.ticker = ticker

    def clean_data(self,df, test=False):
        '''returns a DataFrame without outliers and missing values'''
        df = df.dropna(how='any')
        #df = df.reset_index()
        return df

    def set_pipeline(self, cleaned_data):
        '''returns a pipelined model'''
        data_pipe = Pipeline([('stdscaler', StandardScaler())])
        preproc_pipe = ColumnTransformer(
            [('data', data_pipe,make_column_selector(dtype_include=["int64","float64"]))],remainder="drop")

        pipe = Pipeline([
            ('preproc', preproc_pipe)])

        scaled_data = pipe.fit_transform(cleaned_data)
        return scaled_data,pipe

    def split_timeseries(self, scaled_data, data):
        # Create the training data set
        # Create the scaled training data set
        train_data = scaled_data[0:int(scaled_data.shape[0]*0.80), :]
        y = data[0:int(scaled_data.shape[0]*0.80)]
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(y[i])
            if i<= 61:
                print(x_train)
                print(y_train)
                print()

        # Convert the x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # x_train.shape
        return x_train,y_train

    def split_test_timeseries(self, scaled_data, data):
        # Create the training data set
        # Create the scaled training data set
        test_data = scaled_data[int(scaled_data.shape[0]*0.80):, :]
        y = data[int(scaled_data.shape[0]*0.80):]

        # Split the data into x_train and y_train data sets
        x_test = []
        y_test = []

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            y_test.append(y[i])
            if i<= 61:
                print(x_test)
                print(y_test)
                print()

        # Convert the x_train and y_train to numpy arrays
        x_test, y_test = np.array(x_test), np.array(y_test)
        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test,y_test


    def split_predict(self, scaled_data, data):
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


    # Function to create model, required for KerasClassifier
    def create_model(self, x_train):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1],1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1, activation='linear'))
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # implement train() function
    def train(self, x_train, y_train, model):
        '''returns a trained pipelined model'''
        model.fit(x_train,y_train,epochs=5)
        return model

    # implement evaluate() function
    def evaluate(self, x_test, y_test, model):
        '''returns the value of the RMSE'''
        y_pred = model.predict(x_test)
        mpe = compute_mpe(y_pred, y_test)
        return mpe

    def save_model(self, model, path="model.joblib"):
        """Save the model into a .joblib format"""
        joblib.dump(model, path)
        print("model.joblib saved locally", "green")

    def save_model_to_gcp(self, model, local_model_name="model.joblib"):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        # saving the trained model to disk (which does not really make sense
        # if we are running this code on GCP, because then this file cannot be accessed once the code finished its execution)
        self.save_model(model, path=local_model_name)
        client = storage.Client().bucket(BUCKET_NAME)
        storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
        blob = client.blob(storage_location)
        blob.upload_from_filename(local_model_name)
        print(
            "uploaded model.joblib to gcp cloud storage under \n => {}".format(
                storage_location))


if __name__ == "__main__":
    import datetime
    #define stock code to train the model
    ticker = "INFY.NS"
    trainer = Trainer(ticker=ticker)
    #Get Data
    start_date = (datetime.datetime.now() - datetime.timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    cleaned_data = get_technical(symbol=ticker, start=start_date, end=end_date)
    print(cleaned_data.head(10))
    #Scale the data
    scaled_data, pipe = trainer.set_pipeline(cleaned_data)
    print(scaled_data)
    #Split data in trainning and testing
    x_train, y_train = trainer.split_timeseries(scaled_data, cleaned_data['Close'].to_numpy())
    x_test, y_test = trainer.split_test_timeseries(scaled_data, cleaned_data['Close'].to_numpy())
    #Create Model
    model = trainer.create_model(x_train)
    #Train Model
    model = trainer.train(x_train, y_train, model)
    #Evaluate Model
    mpe = trainer.evaluate(x_test, y_test, model)
    #Print Root Mean Square Error
    print(mpe)
    #Save Model
    trainer.save_model_to_gcp(model, local_model_name=f"{ticker}.joblib")
