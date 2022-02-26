from stockanalysis.database import *
from stockanalysis.param import config
import mysql.connector as connection
from stockanalysis.database import *
from stockanalysis.data import *
from stockanalysis.moneycontrol import *
import datetime
from stockanalysis.moneycontrol import *
from stockanalysis.finviz import *

conn = connection.connect(**config)
mycursor = conn.cursor(dictionary=True)

query = f"SELECT * FROM stocksdb.StocksList;"
mycursor.execute(query)
stock_list = mycursor.fetchall()

for stock in stock_list:
    query = f"""
    SELECT Date FROM stocksdb.raw_news where stock_id={stock['ID']} ORDER BY Date DESC LIMIT 1;
    """
    mycursor.execute(query)
    stock_db_lastdate = mycursor.fetchone()

    if stock['exchange'] =='NASDAQ':
        finviz = FinViz(stock['StockCode'], max_results=10)
        df_news = finviz.get_finviz_news()

    if (stock['exchange'] == 'BSE' or stock['exchange'] == 'NSE'):
        moneycontrol = MoneyControl(stock['moneycontrol_code'], pages=2)
        df_news = moneycontrol.create_news_df()


    if stock_db_lastdate == None:
        df_news = df_news
    else:
        last_date_in_DB = datetime.datetime.strptime(stock_db_lastdate['Date'],'%Y-%m-%d %H:%M:%S').date()
        df_news = df_news.loc[(df_news['date'] > last_date_in_DB)]


    for index, row in df_news.iterrows():

        query = f"""
        INSERT INTO stocksdb.raw_news(ticker,date,title,text,source,url,clean_text,sentiment,stock_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        mycursor.execute(
            query,
            (stock['StockCode'], row.date, row.title, row.text, row.source,
             row.url, row.clean_text, row.sentiment, stock['ID']))
        print(query)

conn.commit()
