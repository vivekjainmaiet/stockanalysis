import re
from textblob import TextBlob
import numpy as np
import pandas as pd
import pandas_ta as pta
from datetime import datetime  # Get the stock quote
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def compute_rmse(y_pred, y_true):
    '''returns root mean square error'''
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def compute_mpe(y_pred, y_true):
    return 1 - abs(y_pred / y_true).mean()


def get_sma(df, period=5, column='Close'):
    '''returns simple moving average of provide column and period'''
    return pta.sma(df[column],length=period)

def get_ema(df, period=10 , column='Close'):
    '''returns simple moving average of provide column and period'''
    return pta.ema(df[column], length=period)

def get_hma(df, period=10 ,column='Close'):
    '''returns simple moving average of provide column and period'''
    return pta.hma(df[column], length=period)

def get_rsi(df,period=14):
    '''returns relative strength index of provided period'''
    return pta.rsi(df['Close'], length = period)

def get_atr(df,period=14):
    '''returns average true range of provided period'''
    return pta.atr(df['High'],df['Low'],df['Close'],length=period)

def get_bband(df,period=20,std=2):
    '''returns Upper , Lower and Middle bolinger band of provided period and std'''
    return pta.bbands(df['Close'],length=period,std=std)

def get_macd(df,fast=12, slow=26, signal=9):
    '''returns Moving average convergence divergence (MACD)'''
    return pta.macd(df['Close'],fast=fast, slow=slow, signal=signal)

def get_adx(df,length=14):
    '''returns ADX of provided period'''
    return pta.adx(df['High'],df['Low'],df['Close'],length=length)

def get_vwap(df):
    '''returns Voumne weighted average'''
    return pta.vwap(df['High'],df['Low'],df['Close'], df['Volume'])


def get_donchian(df, lower_length=20, upper_length=20):
    '''returns Voumne weighted average'''
    return pta.donchian(df['High'],
                        df['Low'],
                        lower_length=20,
                        upper_length=20)

def get_stock_info(ticker):
    '''returns a DataFrame with stock detailed information.'''
    df = pd.DataFrame()
    df = pd.concat([pd.DataFrame([pd.Series(ticker.info.values())]), df], ignore_index=False)
    df.columns =list(ticker.info.keys())
    return df

def isSupport(df,i):
    '''returns True or False for isSupport'''
    support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
    return support

def isResistance(df,i):
    '''returns True or False for isResistance'''
    resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
    return resistance

def isFarFromLevel(df,l,levels):
    '''returns True or False to supress support and registnace duplicate/close lines'''
    s =  np.mean(df['High'] - df['Low'])
    return np.sum([abs(l-x) < s  for x in levels]) == 0

def get_support_registance_levels(df):
    '''returns a DataFrame with stock support and registnace level.'''
    levels = []
    for i in range(2,df.shape[0]-2):
        if isSupport(df,i):
            l = df['Low'][i]
            if isFarFromLevel(df,l,levels):
                levels.append((i,l))
            elif isResistance(df,i):
                l = df['High'][i]
                if isFarFromLevel(df,l,levels):
                    levels.append((i,l))
    return pd.DataFrame(levels, columns=['candle_number','key_level'])


def string_to_sentiment(text):

    return TextBlob(text).sentiment.polarity


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt


def clean_text(df,column='text'):
    '''preprocess and clean text'''

    # remove twitter handles (@user)
    df['clean_text'] = np.vectorize(remove_pattern)(df[column], "@[\w]*")
    # remove special characters, numbers, punctuations
    df['clean_text'] = df['clean_text'].str.replace("[^a-zA-Z#]", " ")
    #removing short words
    df['clean_text'] = df['clean_text'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

    return df


def create_sentiment(df):
    '''Create sentiment from text --> output range [-1,1]'''
    df['sentiment']= df['clean_text'].apply(string_to_sentiment)
    df=df.round(2) #round numbers to 2 decimals
    return df

def lower(text):
    return text.lower()


def deEmojify(text):
    regrex_pattern = re.compile(
        pattern="["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"
        u"\xF0\x9F\xAA\x82"  # flags (iOS)
        "]+",
        flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


chars_to_remove = '€$!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'


def lower(text):

    return text.lower()


def clean_twitter_text(text):

    for punctuation in chars_to_remove:
        text = text.replace(punctuation, ' ')

    return text
