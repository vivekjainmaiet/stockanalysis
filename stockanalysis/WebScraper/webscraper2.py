from types import new_class
import requests
from bs4 import BeautifulSoup
import pandas as pd
from stockanalysis.utils import *


class MoneyControl:

    def __init__(self, ticker, pages=1):
        self.ticker = ticker
        self.pages = pages
        
    