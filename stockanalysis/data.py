import pandas as pd
import yfinance as yf
import numpy as np
import pandas_ta as pta
import tweepy
import requests
from bs4 import BeautifulSoup
from stockanalysis.utils import *
from pandas_datareader import data as pdr
yf.pdr_override()

def get_technical(symbol="INFY.NS",start="2017-01-01", end="2022-04-30"):
    '''returns a DataFrame with stock technical data'''
    df = pdr.get_data_yahoo(symbol, start=start, end=end)
    df.drop(columns=['Adj Close'],inplace=True)

    df['ema12'] = get_ema(df, column='Close', period=12)
    df['ema21'] = get_ema(df, column='Close', period=21)
    df['ema26'] = get_ema(df, column='Close', period=26)
    df['ema34'] = get_ema(df, column='Close', period=34)
    df['ema55'] = get_ema(df, column='Close', period=55)
    df['ema99'] = get_ema(df, column='Close', period=99)
    df['ema200'] = get_ema(df, column='Close', period=200)
    df['hma12'] = get_hma(df, column='Close', period=12)
    df['hma21'] = get_hma(df, column='Close', period=21)
    df['hma26'] = get_hma(df, column='Close', period=26)
    df['hma34'] = get_hma(df, column='Close', period=34)
    df['hma55'] = get_hma(df, column='Close', period=55)
    df['hma99'] = get_hma(df, column='Close', period=99)
    df['hma200'] = get_hma(df, column='Close', period=200)
    df['rsi'] = get_rsi(df, period=14)
    df['atr'] = get_atr(df, period=14)
    df['bb_upper'] = get_bband(df, period=20, std=2)['BBU_20_2.0']
    df['bb_lower'] = get_bband(df, period=20, std=2)['BBL_20_2.0']
    df['macd_signal'] = get_macd(df, fast=12, slow=26,
                                 signal=9)['MACD_12_26_9']
    df['macd_line'] = get_macd(df, fast=12, slow=26, signal=9)['MACDs_12_26_9']
    df['adx'] = get_adx(df, length=14)['ADX_14']
    df['vwap'] = get_vwap(df)
    cleaned_df = clean_data(df)
    return cleaned_df

def get_fundamental():
    '''returns a DataFrame of stock fundamental data'''
    pass

def get_tweets():
    '''returns a DataFrame with tweets realted to stock'''


def get_stock_news():
    '''returns a DataFrame with latest news realted of stock'''
    pass

def get_recommendation():
    '''returns a DataFrame with stock analysis buy sell recommendations'''
    pass

def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    df = df.dropna(how='any')
    df = df.reset_index()
    return df


if __name__ == '__main__':
    from stockanalysis.database import *
    import datetime
    from dateutil.relativedelta import relativedelta

    conn = connection.connect(**config)
    mycursor = conn.cursor(dictionary=True)
    query = f"""
    SELECT Date FROM stocksdb.raw_technical where Stock_id=1 ORDER BY Date DESC LIMIT 1;
    ;"""
    mycursor.execute(query)
    stock_db_lastdate = mycursor.fetchone()

    if stock_db_lastdate == None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=5 * 365)).strftime("%Y-%m-%d")
        df = get_technical(symbol="INFY.NS", start=start_date, end=end_date)
        print(df)
    else:
        date_time_str = stock_db_lastdate['Date'].split()[0]
        date_time_obj = datetime.strptime(date_time_str, '%Y/%m/%d')
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = date_time_obj + datetime.timedelta(days=1)
        df = get_technical(symbol="INFY.NS", start="2017-01-01", end="2022-04-30")
        print(df)

    #print(stock_db_lastdate['Date'].split()[0])
    #df = get_technical()
    #print(df)
    #cleaned_df = cleaned_df
    #cleaned_df.to_csv("raw_technical.csv", sep='\t', encoding='utf-8')
    #query = ("INSERT INTO stocksdb.TechnicalData (Date,StockID) "
    #"VALUES (%s, 1)")
    #db.SaveDFToTable(query, cleaned_df)
