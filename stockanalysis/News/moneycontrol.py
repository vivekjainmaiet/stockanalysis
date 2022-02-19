from types import new_class
import requests
from bs4 import BeautifulSoup
import pandas as pd
from stockanalysis.utils import *


class MoneyControl:

    def __init__(self, ticker, max_results=100):
        self.ticker = ticker
        self.max_results = max_results
        self.url = f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={ticker}&durationType=M&duration=6"


    def get_nse_sentiments(self, recommendation=False, news=False):

        #url = "https://www.moneycontrol.com/company-article/    /news/{ticker}"
        request = requests.get(self.url)

        parser = BeautifulSoup(request.text, "html.parser")
        recmd_html = parser.find_all('div', {'class': 'MT15 PT10 PB10'})

        recommendations = []
        news_list = []

        for i in range(0, len(recmd_html)):
            if "recommendations" in recmd_html[i].find_all(
                    'a', {'class': 'arial11_summ'})[0]['href']:
                recommendations.append({
                    'ticker':
                    self.ticker,
                    'date':
                    recmd_html[i].text.split('\n')[5].split('|')[1].strip(),
                    'title':
                    recmd_html[i].text.split('\n')[4].strip(),
                    'text':
                    recmd_html[i].text.split('\n')[6].strip(),
                    'source':
                    recmd_html[i].text.split('\n')[5].split('|')[2].split(':')
                    [1].strip(),
                    'url':
                    recmd_html[i].find_all('a',
                                        {'class': 'arial11_summ'})[0]['href'],
                    'advice':
                    recmd_html[i].text.split('\n')[4].strip().lower().split()[0],
                    'target':
                    int(recmd_html[i].text.split('\n')[4].strip().split(
                        "target of Rs ", 1)[1].split(":", 1)[0].strip()),
                    'analyst':
                    recmd_html[i].text.split('\n')[4].strip().split(
                        "target of Rs ", 1)[1].split(":", 1)[1].strip()
                })
            else:
                news_list.append({
                    'ticker':
                    self.ticker,
                    'date':
                    recmd_html[i].text.split('\n')[5].split('|')[1].strip(),
                    'title':
                    recmd_html[i].text.split('\n')[4].strip(),
                    'text':
                    recmd_html[i].text.split('\n')[6].strip(),
                    'source':
                    recmd_html[i].text.split('\n')[5].split('|')[2].split(':')
                    [1].strip(),
                    'url':
                    recmd_html[i].find_all(
                        'a', {'class': 'arial11_summ'})[0]['href']
                })

        df_recommendations = create_sentiment(clean_text(pd.DataFrame(recommendations), column="title"))
        df_news = create_sentiment(clean_text(pd.DataFrame(news_list)))
        if news and recommendation:
            return df_recommendations, df_news
        elif recommendation:
            return df_recommendations
        else:
            return df_news


if __name__ == "__main__":
    moneycontrol = MoneyControl('TCS', max_results=10)
    df = moneycontrol.get_nse_sentiments(recommendation=True)
    df1=moneycontrol.get_nse_sentiments(news=True)
    df2 = moneycontrol.get_nse_sentiments(recommendation=True,news=True)
    breakpoint()
