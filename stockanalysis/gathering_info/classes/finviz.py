from pprint import pprint
import finvizfinance
import finvizfinance.screener
from finvizfinance.quote import finvizfinance
from stockanalysis.utils import *
import re
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen

pd.set_option('display.max_colwidth', 25)


class FinViz:

    def __init__(self, ticker, max_results=100):
        # input ticker sample want to obtain news for
        self.ticker = ticker
        self.max_results = max_results

    def get_finviz_news(self):
        stock = finvizfinance(self.ticker)
        news = stock.ticker_news()
        news['ticker'] = self.ticker
        news['text'] = "NA"
        news['source'] = news['Link']
        news['source'] = news['source'].apply(lambda x: x.split("//")[1].split("/")[0])
        df_news = create_sentiment(clean_text(news,column='Title'))
        df_news = df_news.rename(columns={"Date": "date","Title":"title","Link":"url"})
        return df_news

    def get_finviz_recommendations(self):
        # Set up scraper
        url = ("http://finviz.com/quote.ashx?t=" + self.ticker.lower())
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        html = soup(webpage, "html.parser")
        list_recommendation = []
        for recommendation in html.find_all(
                'tr', {'class': 'body-table-rating-upgrade'}):
            list_recommendation.append(recommendation.text)

        for recommendation in html.find_all(
                'tr', {'class': 'body-table-rating-neutral'}):
            list_recommendation.append(recommendation.text)

            mystring = list_recommendation[0].split("\n")[1]

        recommendation_dict = {
            'ticker': [],
            'date': [],
            'title': [],
            'text': [],
            'source': [],
            'url': [],
            'advice': [],
            'target': [],
            'analyst': [],
            'sentiment': [],
            'clean_text': []
        }
        mo = re.match('.+([0-9])[^0-9]*$', mystring)

        for value in list_recommendation:
            recommendation_dict['ticker'].append(self.ticker)
            recommendation_dict['date'].append(list_recommendation[0].split("\n")[1][:mo.start(1) + 1])
            recommendation_dict['title'].append("NULL")
            recommendation_dict['text'].append("NULL")
            recommendation_dict['sentiment'].append(0)
            recommendation_dict['clean_text'].append("NULL")
            recommendation_dict['url'].append(f"https://finviz.com/quote.ashx?t=self.ticker{self.ticker}")
            recommendation_dict['source'].append("finviz")
            recommendation_dict['advice'].append(list_recommendation[0].split("\n")[1][mo.start(1) + 1:])
            recommendation_dict['analyst'].append(value.split("\n")[1])
            recommendation_dict['target'].append(value.split("\n")[3])


        return pd.DataFrame(recommendation_dict)


if __name__ == "__main__":
    finviz = FinViz('AAPL', max_results=10)
    df_news = finviz.get_finviz_news()
    print(df_news.columns)
    df_recommendation = finviz.get_finviz_recommendations()
    print(df_recommendation)
