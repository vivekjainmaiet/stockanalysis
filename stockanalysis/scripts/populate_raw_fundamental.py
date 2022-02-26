from stockanalysis.database import *
from stockanalysis.param import config
from stockanalysis.data import *
from stockanalysis.moneycontrol import *
from stockanalysis.finviz import *
import mysql.connector as connection
from database import *

conn = connection.connect(**config)
mycursor = conn.cursor(dictionary=True)

query = f"SELECT * FROM stocksdb.StocksList;"
mycursor.execute(query)
stock_list = mycursor.fetchall()

for stock in stock_list:
    print(stock['exchange'])
    if stock['exchange'] =='NASDAQ':
        finviz = FinViz(stock['StockCode'], max_results=10)
        df_news = finviz.get_finviz_news()
        print(df_news.columns)

    if (stock['exchange'] == 'BSE' or stock['exchange'] == 'NSE'):
        moneycontrol = MoneyControl(stock['moneycontrol_code'], pages=2)
        df_news = moneycontrol.create_news_df()
        print(df_news.columns)

    for index, row in df_news.iterrows():

        query = f"""
        INSERT INTO stocksdb.raw_news(ticker,date,title,text,source,url,clean_text,sentiment,stock_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        mycursor.execute(
            query, (stock['StockCode'], row.date, row.title, row.text, row.source,
                    row.url, row.clean_text, row.sentiment, stock['ID']))
        print(query)

conn.commit()
