import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta

import pandas_ta as pta


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

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
        data_pipe = Pipeline([('stdscaler', MinMaxScaler())])
        preproc_pipe = ColumnTransformer(
            [('data', data_pipe,make_column_selector(dtype_include=["int64","float64"]))],remainder="drop")

        pipe = Pipeline([('preproc', preproc_pipe)])
        scaled_data = pipe.fit_transform(cleaned_data)

        return scaled_data,pipe

    def split_timeseries(self, scaled_data, data, sequence_size=21):
        # Create the training data set
        # Create the scaled training data set
        train_data = scaled_data
        y = data
        #breakpoint()
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []

        for i in range(sequence_size, len(train_data)):
            x_train.append(train_data[i - sequence_size:i, 0])
            y_train.append(y[i])
            if i <= sequence_size+1:
                print(x_train)
                print(y_train)
                print()

        # Convert the x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # x_train.shape
        return x_train,y_train

    # Function to create model, required for KerasClassifier
    def create_model(self, x_train):
        #model = Sequential()
        #model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1],1)))
        #model.add(LSTM(64, return_sequences=False))
        #model.add(Dense(25))
        #model.add(Dense(1, activation='linear'))
        # Compile the model
        #model.compile(optimizer='adam', loss='mean_squared_error')
        #return model
        model = Sequential()
        model.add(LSTM(50,return_sequences=True,activation='tanh',input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',optimizer=RMSprop(learning_rate=0.03), metrics=['mae'])
        return model

    # implement train() function
    def train(self, X_train, y_train, model):
        '''returns a trained pipelined model'''
        es = EarlyStopping(patience=PATIENCE, restore_best_weights=True)

        train_size = 0.8
        train_sample = int(train_size * X_train.shape[0])
        X_train, y_train, X_val, y_val = X_train[:train_sample], y_train[:train_sample], X_train[train_sample:], y_train[train_sample:]

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=es,batch_size=128)
        #model.fit(x_train,y_train,epochs=100,)
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
    cleaned_data = get_technical(symbol=ticker, start=start_date,
                                 end=end_date)[COLUMNS]
    print(cleaned_data.head(10))

    # Time Serie split
    len_ = int(0.8*cleaned_data.shape[0])
    df_train_dataset = cleaned_data[:len_]
    df_test_dataset = cleaned_data[len_:]
    #breakpoint()
    #Scale the data
    scaled_data_train, pipe_train = trainer.set_pipeline(df_train_dataset)
    scaled_data_test = pipe_train.transform(df_test_dataset)
    print(scaled_data_train, scaled_data_test)
    #Split data in trainning and testing
    y_train=df_train_dataset['Close'].to_numpy()
    y_test=df_test_dataset['Close'].to_numpy()
    x_train, y_train = trainer.split_timeseries(scaled_data_train,
                                                np.log(y_train),
                                                sequence_size=SEQUENCE_SIZE)
    x_test, y_test = trainer.split_timeseries(scaled_data_test,
                                              np.log(y_test),
                                              sequence_size=SEQUENCE_SIZE)

    #breakpoint()
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
