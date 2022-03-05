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

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf


import joblib
from google.cloud import storage

from stockanalysis.data import *
from stockanalysis.encoder import *
from stockanalysis.utils import *
from stockanalysis.param import *

from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

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

    def split_timeseries(self,
                         scaled_data,
                         data,
                         sequence_size=21,
                         y_len=Y_LEN):
        # Create the training data set
        # Create the scaled training data set
        train_data = scaled_data
        y = data
        #breakpoint()
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []

        for i in range(sequence_size, len(train_data)):
            if len(y[i:i+y_len]) < y_len:
                break

            x_train.append(train_data[i - sequence_size:i, :])
            y_train.append(y[i:i+y_len])

            if i <= sequence_size+1:
                # print(x_train)
                # print(y_train)
                print('i <= sequence_size+1')


        # Convert the x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        # Reshape the data

        #y_train = np.reshape(y_train, (y_train.shape[0],Y_LEN))

        # x_train.shape
        return x_train,y_train

    # Function to create model, required for KerasClassifier
    def create_model(self, X_train, y_train):
        #model = Sequential()
        #model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1],1)))
        #model.add(LSTM(64, return_sequences=False))
        #model.add(Dense(25))
        #model.add(Dense(1, activation='linear'))
        # Compile the model
        #model.compile(optimizer='adam', loss='mean_squared_error')
        #return model
        # breakpoint()
        tf.random.set_seed(30)
        # np.random.seed(42)
        normalizer = Normalization() # Instantiate a "normalizer" layer
        normalizer.adapt(X_train) # "Fit" it on the train set
        model = Sequential()
        model.add(normalizer)
        model.add(layers.LSTM(20, return_sequences=False, input_shape=(X_train.shape[1],X_train.shape[2])))
        # model.add(layers.LSTM(3, return_sequences=True, recurrent_dropout=0.3))
        # model.add(layers.LSTM(5))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(y_train.shape[1], activation='relu'))
        model.add(layers.Dense(Y_LEN, activation='linear'))
        model.compile(loss='mse',optimizer=RMSprop(learning_rate=0.0075), metrics=['mae', 'mape'])
        # model.compile(loss='mse',optimizer='rmsprop', metrics=['mae', 'mape'])
        print(model.summary())
        print(X_train.shape)

        return model

    # implement train() function
    def train(self, X_train, y_train, model):
        '''returns a trained pipelined model'''
        es = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
        #breakpoint()
        train_size = 0.9
        train_sample = int(train_size * X_train.shape[0])
        X_train, y_train, X_val, y_val = X_train[:train_sample], y_train[:train_sample], X_train[train_sample:], y_train[train_sample:]

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, callbacks=es,batch_size=128)

        #model.fit(x_train,y_train,epochs=100,)
        return model

    # implement evaluate() function
    def evaluate(self, x_test, y_test, model):
        '''returns the value of the RMSE'''
        y_pred = model.predict(x_test)
        print(y_pred.shape)
        mpe = compute_mpe(y_pred, y_test)

        residuos = y_test - y_pred
        r2_score_ = r2_score(y_test, y_pred)
        rmse = (residuos ** 2).mean() ** 0.5
        mean_absolute_error_ = abs(residuos).mean()
        df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred[:,0]})
        df = df.apply(np.exp)
        df['residuos'] = df['y_true'] - df['y_pred']
        df['y_true'].plot(label = 'y_true')
        df['y_pred'].plot(label = 'y_pred')
        plt.show()
        df['residuos'].plot(label = 'residuos')
        plt.show()
        print(df.mean())
        print(df.corr())
        print(df.head())
        print(f'''
              r2_score = f{r2_score_},
              rmse = f{rmse}
              mean_absolute_error = f{mean_absolute_error_}
              ''')
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
    len_ = int(0.80*cleaned_data.shape[0])
    df_train_dataset = cleaned_data[:len_]
    df_test_dataset = cleaned_data[len_:]
    #breakpoint()
    #Scale the data
    # scaled_data_train, pipe_train = trainer.set_pipeline(df_train_dataset)
    # scaled_data_test = pipe_train.transform(df_test_dataset)
    # print(scaled_data_train, scaled_data_test)
    #Split data in trainning and testing

    #Test
    X_train_dataset = df_train_dataset.to_numpy()
    X_test_dataset = df_test_dataset.to_numpy()



    print('TYPE_Y : ',  TYPE_Y)
    if TYPE_Y == 'log':
        y_train=np.log(df_train_dataset['Close'].to_numpy())
        y_test=np.log(df_test_dataset['Close'].to_numpy())
    elif TYPE_Y == 'pct_change':
        y_train=df_train_dataset['Close'].pct_change().to_numpy()
        y_test=df_test_dataset['Close'].pct_change().to_numpy()
    elif TYPE_Y == 'diff':
        y_train=df_train_dataset['Close'].diff().to_numpy()
        y_test=df_test_dataset['Close'].diff().to_numpy()
    else:
        y_train=df_train_dataset['Close'].to_numpy()
        y_test=df_test_dataset['Close'].to_numpy()


    stationary_train = adfuller(y_train[~np.isnan(y_train)])[1]
    stationary_test = adfuller(y_test[~np.isnan(y_test)])[1]

    print(f'''
          adfuller_train: {stationary_train},
          adfuller_test: {stationary_test}
          ''')



    x_train, y_train = trainer.split_timeseries(X_train_dataset,
                                                y_train,
                                                sequence_size=SEQUENCE_SIZE)
    x_test, y_test = trainer.split_timeseries(X_test_dataset,
                                              y_test,
                                              sequence_size=SEQUENCE_SIZE)

    # breakpoint()
    #Create Model
    model = trainer.create_model(x_train, y_train)
    #Train Model
    model = trainer.train(x_train, y_train, model)
    #Evaluate Model
    mpe = trainer.evaluate(x_test, y_test, model)
    #Print Root Mean Square Error
    #breakpoint()
    print(mpe)
    #Save Model
    trainer.save_model_to_gcp(model, local_model_name=f"{ticker}.joblib")
