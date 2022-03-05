from nis import match
from stockanalysis.database import *
from stockanalysis.param import config
import mysql.connector as connection
from stockanalysis.data import *
from stockanalysis.gathering_info.classes.moneycontrol import *
import datetime

conn = connection.connect(**config)
mycursor = conn.cursor(dictionary=True)

query = f"SELECT * FROM stocksdb.StocksList;"
mycursor.execute(query)
stock_list = mycursor.fetchall()



for stock in stock_list:
    update = True

    query = f"""
    SELECT Date FROM stocksdb.raw_technical where Stock_id={stock['ID']} ORDER BY ID DESC LIMIT 1;
    """
    mycursor.execute(query)
    stock_db_lastdate = mycursor.fetchone()

    start_date = (datetime.datetime.now() - datetime.timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    if stock_db_lastdate == None:
        df = get_technical(symbol=stock['yahoo_code'], start=start_date, end=end_date).round(2)
    else:
        df = get_technical(symbol=stock['yahoo_code'], start=start_date, end=end_date).round(2)
        index = df.index[df['Date'] == stock_db_lastdate['Date']][0]

        if df.shape[0] == index + 1:
            print(f"{stock['StockName']} technical data is already upto date.")
            update = False
        else:
            df = df.tail(df.shape[0] - (index+1))



    if update:
        for index,row in df.iterrows():
            query = f"""
                INSERT INTO stocksdb.raw_technical(Date,Stock_id,Open, High, Low, Close, Volume, ema12,
                ema21, ema26,ema34, ema55, ema99, ema200, hma12, hma21, hma26, hma34,hma55, hma99, hma200,
                rsi, atr, bb_upper, bb_lower,macd_signal, macd_line, adx, vwap)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """
            mycursor.execute(query,(str(row.Date), stock['ID'], row.Open, row.High, row.Low,
                row.Close, row.Volume, row.ema12, row.ema21, row.ema26, row.ema34,
                row.ema55, row.ema99, row.ema200, row.hma12, row.hma21, row.hma26,
                row.hma34, row.hma55, row.hma99, row.hma200, row.rsi, row.atr,
                row.bb_upper, row.bb_lower, row.macd_signal, row.macd_line,
                row.adx, row.vwap))
        conn.commit()
