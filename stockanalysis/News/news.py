import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
from stockanalysis.utils import *
pd.set_option('display.max_colwidth', 25)
import finvizfinance
import finvizfinance.screener
from finvizfinance.quote import finvizfinance


def get_fin_news(ticker='AAPL'):
    # obtain news per ticker
    stock = finvizfinance(ticker)
    news = stock.ticker_news()
    news['Ticker'] = ticker
    news['Date'] = pd.to_datetime(news['Date'])
    # save the news
    # news.to_csv('/Users/yingxu/stockanalysis/notebooks/news')
    return news

df = get_fin_news('AAPL')
df = clean_text(df, column="Title")
df = create_sentiment(df)
print(round(df['sentiment'].mean(),2))
