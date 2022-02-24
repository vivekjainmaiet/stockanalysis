from stockanalysis.database import *
from stockanalysis.param import config
from stockanalysis.data import *
from stockanalysis.moneycontrol import *
import mysql.connector as connection
from database import *
conn = connection.connect(**config)
mycursor = conn.cursor(dictionary=True)

#query = f"SELECT * FROM stocksdb.StocksList;"
#mycursor.execute(query)
#stock_list = mycursor.fetchall()
stock_list = [{
    'ID': 1,
    'StockName': 'Apple Inc.',
    'StockCode': 'AAPL',
    'exchange': 'NASDAQ',
    'Currency': 'USD',
    'yahoo_code': 'AAPL'
}]
#print(stock_list)
#for stock in stock_list:
#Fetch technical and store in database
#print(get_technical(symbol=stock['yahoo_code'], period='1y').tail(10))
#Fetch news , recommendation and store in database for BSE stocks
#moneycontrol = MoneyControl(stock['StockCode'], pages=2)
#if stock['exchange'] == 'BSE':
#df_recommendation= moneycontrol.create_recommendation_df()
#print(df_recommendation['title'])
#df_news = moneycontrol.create_news_df()
#print(df_news['title'])

#Fetch news , recommendation and store in database for NASDAQ stocks

df = get_technical(symbol='AAPL', period='1y').tail(1)
print(df.columns)
