import numpy as np
import pandas_ta as pta
import pandas as pd

def compute_rmse(y_pred, y_true):
    '''returns root mean square error'''
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def get_sma(df,column='Close',period=5):
    '''returns simple moving average of provide column and period'''
    return pta.sma(df[column],length=period)

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

def get_stock_info(ticker):
    '''returns a DataFrame with stock detailed information.'''
    df = pd.DataFrame()
    df = pd.concat([pd.DataFrame([pd.Series(ticker.info.values())]), df], ignore_index=False)
    df.columns =list(ticker.info.keys())
    return df
