from stockanalysis.database import *
from stockanalysis.param import config
import mysql.connector as connection
from stockanalysis.database import *
from stockanalysis.data import *
from stockanalysis.gathering_info.classes.moneycontrol import *
import datetime
from stockanalysis.gathering_info.classes.finviz import *

conn = connection.connect(**config)
mycursor = conn.cursor(dictionary=True)

query = f"SELECT * FROM stocksdb.StocksList;"
mycursor.execute(query)
stock_list = mycursor.fetchall()


for stock in stock_list:
    print(stock)
    update = True

    if stock['exchange'] =='NASDAQ':
        query = f"""
                SELECT Date FROM stocksdb.raw_news where stock_id={stock['ID']} ORDER BY date DESC LIMIT 1;
                """
        mycursor.execute(query)
        stock_db_lastdate = mycursor.fetchone()
        finviz = FinViz(stock['StockCode'])
        df_news = finviz.get_finviz_news()
        if stock_db_lastdate == None:
            df_news = df_news
        else:
            if len(df_news.index[df_news['date'] == stock_db_lastdate['Date']]) != 0:
                index = df_news.index[df_news['date'] == stock_db_lastdate['Date']][0]
                if (index == 0):
                    print(f"{stock['StockCode']} news is already upto date.")
                    update = False
                else:
                    df_news = df_news.head(index)
            else:
                print("Date time did not match")


    if (stock['exchange'] == 'BSE' or stock['exchange'] == 'NSE'):

        query = f"""
        SELECT Date FROM stocksdb.raw_news where stock_id={stock['ID']} ORDER BY ID ASC LIMIT 1;
        """
        mycursor.execute(query)
        stock_db_lastdate = mycursor.fetchone()

        moneycontrol = MoneyControl(stock['moneycontrol_code'], pages=2)
        df_news = moneycontrol.create_news_df()
        if stock_db_lastdate == None:
            df_news = df_news
        else:
            index = df_news.index[df_news['date'] == stock_db_lastdate['Date']][0]
            #First row
            if index == 0:
                print(f"{stock['StockCode']} news is already upto date.")
                update = False
            else:
                df_news = df_news.head(index)


    if update:
        for index, row in df_news.iterrows():

            query = f"""
            INSERT INTO stocksdb.raw_news(ticker,date,title,text,source,url,clean_text,sentiment,stock_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """
            mycursor.execute(
                query,
                (stock['StockCode'], str(row.date), row.title, row.text, row.source,
                row.url, row.clean_text, row.sentiment, stock['ID']))
        conn.commit()
