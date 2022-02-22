from types import new_class
import requests
from bs4 import BeautifulSoup
import pandas as pd
from stockanalysis.utils import *


def get_nse_sentiments(ticker='TCS', recommendation=False, news=False):
    url = f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={ticker}&durationType=M&duration=6"
    #url = "https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=IT&durationType=M&duration=6"
    request = requests.get(url)

    parser = BeautifulSoup(request.text, "html.parser")
    recmd_html = parser.find_all('div', {'class': 'MT15 PT10 PB10'})

    recommendations = []
    news = []

    for i in range(0, len(recmd_html)):
        if "recommendations" in recmd_html[i].find_all('a', {'class': 'arial11_summ'})[0]['href']:
            recommendations.append({
                'ticker':
                ticker,
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
                recmd_html[i].find_all('a', {'class': 'arial11_summ'})[0]['href'],
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
            news.append({
                'ticker':
                ticker,
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
                recmd_html[i].find_all('a', {'class': 'arial11_summ'})[0]['href']
            })

    df_recommendations = pd.DataFrame(recommendations)
    df_news = pd.DataFrame(news)
    if recommendation and news:
        return df_recommendations, df_news
    elif recommendation:
        return df_recommendations
    else:
        return df_news


if __name__ == "__main__":

    recommendation = get_nse_sentiments(ticker='TCS',recommendation=True, news=True)
    news = get_nse_sentiments(ticker='TCS', recommendation=False, news=True)
    df_recommendation = clean_text(recommendation, column="title")
    df_recommendation = create_sentiment(df_recommendation)
    df_news = clean_text(news, column="title")
    df_news = create_sentiment(df_news)
    print(df_news)
