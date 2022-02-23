from types import new_class
import requests
from bs4 import BeautifulSoup
import pandas as pd
from stockanalysis.utils import *


class MoneyControl:

    def __init__(self, ticker, pages=1):
        self.ticker = ticker
        self.pages = pages

    def fetch_page(self,page):
        print(f"scraping page {page + 1}")
        response = requests.get(
            f'https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={self.ticker}&scat=&pageno={page + 1}&next=0&durationType=M&Year=&duration=6&news_type=',
            headers={"Accept-Language": "en-US"})
        soup = BeautifulSoup(response.content, "html.parser")
        return soup

    def add_recommendation_to_dict(self, soup,dict):
        for recmd_html in soup.find_all('div', {'class': 'MT15 PT10 PB10'}):
            if "recommendations" in recmd_html.find_all('a', {'class': 'arial11_summ'})[0]['href']:
                dict['ticker'].append(self.ticker)
                dict['date'].append(recmd_html.text.split('\n')[5].split('|')[1].strip())
                dict['title'].append(recmd_html.text.split('\n')[4].strip())
                dict['text'].append(recmd_html.text.split('\n')[6].strip())
                dict['source'].append(recmd_html.text.split('\n')[5].split('|')[2].split(':')[1].strip())
                dict['url'].append(recmd_html.find_all('a', {'class': 'arial11_summ'})[0]['href'])
                dict['advice'].append(recmd_html.text.split('\n')[4].strip().lower().split()[0])
                dict['target'].append(int(recmd_html.text.split('\n')[4].strip().split("target of Rs ", 1)[1].split(":", 1)[0].strip()))
                dict['analyst'].append(recmd_html.text.split('\n')[4].strip().split("target of Rs ", 1)[1].split(":", 1)[1].strip())

    def create_recommendation_df(self):
        recommendation_dict = {'ticker': [], 'date': [], 'title': [],'text':[],'source':[],'url':[],'advice':[],'target':[],'analyst':[]}
        for page in range(self.pages):
            soup = self.fetch_page(page)
            self.add_recommendation_to_dict(soup,recommendation_dict)
        return pd.DataFrame.from_dict(recommendation_dict)

    def add_news_to_dict(self,soup, dict):
        for recmd_html in soup.find_all('div', {'class': 'MT15 PT10 PB10'}):
            if not("recommendations" in recmd_html.find_all('a', {'class': 'arial11_summ'})[0]['href']):
                dict['ticker'].append(self.ticker)
                dict['date'].append(recmd_html.text.split('\n')[5].split('|')[1].strip())
                dict['title'].append(recmd_html.text.split('\n')[4].strip())
                dict['text'].append(recmd_html.text.split('\n')[6].strip())
                dict['source'].append(recmd_html.text.split('\n')[5].split('|')[2].split(':')[1].strip())
                dict['url'].append(recmd_html.find_all('a', {'class': 'arial11_summ'})[0]['href'])

    def create_news_df(self):
        news_dict = {'ticker': [], 'date': [], 'title': [],'text':[],'source':[],'url':[]}
        for page in range(self.pages):
            soup = self.fetch_page(page)
            self.add_news_to_dict(soup, news_dict)
        return pd.DataFrame.from_dict(news_dict)

if __name__ == "__main__":
    moneycontrol = MoneyControl('TCS', pages=2)
    #df_recommendation= moneycontrol.create_recommendation_df()
    #print(df_recommendation)
    df_news = moneycontrol.create_news_df()
    print(df_news['title'])
