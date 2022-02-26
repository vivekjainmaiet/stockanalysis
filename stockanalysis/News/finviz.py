from pprint import pprint
import finvizfinance
import finvizfinance.screener
from finvizfinance.quote import finvizfinance
from stockanalysis.utils import *

class FinViz:

    def __init__(self, ticker, max_results=100):
        # input ticker sample want to obtain news for
        self.ticker = ticker
        self.max_results = max_results

    def get_finviz_news(self):
        stock = finvizfinance(self.ticker)
        news = stock.ticker_news()
        news['ticker'] = self.ticker
        news['source'] = news['Link']
        news['source'] = news['source'].apply(lambda x: x.split("//")[1].split("/")[0])
        df_news = create_sentiment(clean_text(news,column='Title'))
        df_news = df_news.rename(columns={"Date": "date","Title":"title","Link":"url"})
        return df_news


if __name__ == "__main__":
    finviz = FinViz('AAPL', max_results=10)
    df = finviz.get_finviz_news()
    print(df)
