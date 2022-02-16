import pandas as pd
import yfinance
import numpy as np
import pandas_ta as pta
import tweepy
import requests
from bs4 import BeautifulSoup

def get_technical():
    '''returns a DataFrame with stock technical data'''
    pass

def get_fundamental():
    '''returns a DataFrame of stock fundamental data'''
    pass

def get_tweets():
    '''returns a DataFrame with tweets realted to stock'''
    pass

def get_stock_news():
    '''returns a DataFrame with latest news realted of stock'''
    pass

def get_recommendation():
    '''returns a DataFrame with stock analysis buy sell recommendations'''
    pass

def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    df = df.dropna(how='any')
    return df


if __name__ == '__main__':
    df = get_technical()
    cleaned_df = clean_data(df)
