import pandas as pd
import yfinance
import numpy as np
import pandas_ta as pta
import tweepy
import requests
from bs4 import BeautifulSoup
from stockanalysis.utils import *
from stockanalysis.data import *
from stockanalysis.scraper import *
import re
from textblob import TextBlob


df = get_technical()
cleaned_df = clean_data(df)
print(cleaned_df)
