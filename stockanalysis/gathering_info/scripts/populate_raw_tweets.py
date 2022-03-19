import mysql.connector as connection
from stockanalysis import param
from stockanalysis.gathering_info.classes.twitter import *
from stockanalysis.param import *

conn = connection.connect(**config)
mycursor = conn.cursor(dictionary=True)

query = f"SELECT * FROM stocksdb.StocksList;"
mycursor.execute(query)
stock_list = mycursor.fetchall()


for stock in stock_list:
    update = True

    query = f"""
    SELECT Date FROM stocksdb.twitter_sentiment where stock_id={stock['ID']} ORDER BY ID DESC LIMIT 1;
    """
    mycursor.execute(query)
    stock_db_lastdate = mycursor.fetchone()

    scraper = Scraper(stock['twitter_code'], 3000)
    scraper.get_tweets()
    scraper.preprocess_tweets()
    scraper.create_sentiment()
    df_tweets = scraper.save_df()



    if stock_db_lastdate == None:
        df_tweets = df_tweets
        df_tweets['stock_id'] = stock['ID']
    else:
        df_tweets['stock_id'] = stock['ID']

        #breakpoint()
        if df_tweets['Date'][0] == stock_db_lastdate['Date']:
            print(f"{stock['StockName']} tweets data is already upto date.")
            update = False
        else:
            df_tweets = df_tweets

    for index, row in df_tweets.iterrows():
        print(row)
        query = f"""
        INSERT INTO twitter_sentiment(Date,stock_id,neg,pos)
            VALUES (%s,%s,%s,%s);
        """
        mycursor.execute(query, (row.Date, row.stock_id, row.neg, row.pos))

    if update :
        conn.commit()
    print("Done")
