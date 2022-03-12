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
    update = True
    query = f"""
    SELECT Date FROM stocksdb.raw_recommendation where stock_id={stock['ID']} ORDER BY Date DESC LIMIT 1;
    """
    mycursor.execute(query)
    stock_db_lastdate = mycursor.fetchone()
    if stock['exchange'] == 'NASDAQ':
        finviz = FinViz(stock['StockCode'], max_results=10)
        df_recommendation= finviz.get_finviz_recommendations()
        print(df_recommendation)


        if stock_db_lastdate == None:
            df_recommendation = df_recommendation
        else:
            last_date_in_DB = stock_db_lastdate['Date']
            #breakpoint()
            index = df_recommendation.index[df_recommendation['date'] ==last_date_in_DB][0]

            #First row
            if index == 0:
                print(f"{stock['StockName']} recommendation is already upto date.")
                update = False
            else:
                df_recommendation = df_recommendation.head(index)

    if (stock['exchange'] == 'BSE' or stock['exchange'] == 'NSE'):
        moneycontrol = MoneyControl(stock['moneycontrol_code'], pages=2)
        df_recommendation = moneycontrol.create_recommendation_df()

        if stock_db_lastdate == None:
            df_recommendation = df_recommendation
        else:
            last_date_in_DB = stock_db_lastdate['Date']
            index = df_recommendation.index[df_recommendation['date'] == last_date_in_DB][0]

            #First row
            if index == 0:
                print(f"{stock['StockName']} recommendation is already upto date.")
                update = False
            else:
                df_recommendation = df_recommendation.head(index)


    if update:
        for index, row in df_recommendation.iterrows():

            query = f"""
            INSERT INTO stocksdb.raw_recommendation(ticker,date,title,text,source,url,clean_text,sentiment,stock_id,advice,target,analyst)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """
            mycursor.execute(
                query, (stock['StockCode'], row.date, row.title, row.text,
                        row.source, row.url, row.clean_text, row.sentiment,
                        stock['ID'], row.advice, row.target, row.analyst))
        conn.commit()
        print(f"Adding stock {stock['StockCode']} data in databse.")

print("Done")
