from stockanalysis.utils import *
from email.charset import Charset
from stockanalysis.twitter import *
from stockanalysis.param import config
import mysql.connector as connection

conn = connection.connect(**config)
mycursor = conn.cursor(dictionary=True)

query = f"SELECT * FROM stocksdb.StocksList;"
mycursor.execute(query)
stock_list = mycursor.fetchall()


for stock in stock_list:
    update = True

    query = f"""
    SELECT created_at FROM stocksdb.raw_tweets where stock_id={stock['ID']} ORDER BY ID DESC LIMIT 1;
    """
    mycursor.execute(query)
    stock_db_lastdate = mycursor.fetchone()

    scraper = Scraper(stock['StockCode'], 10)
    scraper.get_tweets()
    scraper.preprocess_tweets()
    df_tweets= scraper.create_sentiment()

    if stock_db_lastdate == None:
        df_tweets = df_tweets
    else:
        index = df_tweets.index[df_tweets['created_at'] == stock_db_lastdate['created_at']][0]
        #Last line of dataframe
        if df_tweets.shape[0] == index + 1:
            print(f"{stock['StockName']} tweets data is already upto date.")
            update = False
        else:
            df_tweets = df_tweets.tail(df_tweets.shape[0] - (index + 1))
            print(df_tweets)

    for index, row in df_tweets.iterrows():
        print(row.text)
        query = f"""
        INSERT INTO stocksdb.raw_tweets(Stock_id,text,created_at,clean_text,sentiment)
        VALUES (%s,%s,%s,%s,%s);
        """
        mycursor.execute(query, (stock['ID'], deEmojify(
            row.text), str(row.created_at), row.clean_text, row.sentiment))
    if update :
        conn.commit()
    print("Done")
