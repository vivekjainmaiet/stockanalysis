from stockanalysis.tikcertape import *
import mysql.connector as connection
from stockanalysis.param import *

conn = connection.connect(**config)
mycursor = conn.cursor(dictionary=True)

query = f"SELECT * FROM stocksdb.StocksList;"
mycursor.execute(query)
stock_list = mycursor.fetchall()

for stock in stock_list:
    print(stock)
    update = True

    query = f"""
    SELECT Financial_Year FROM stocksdb.raw_fundamental where stock_id={stock['ID']} ORDER BY ID DESC LIMIT 1;
    """
    mycursor.execute(query)
    stock_lastfinancial_year = mycursor.fetchone()

    if stock['exchange'] =='NASDAQ':
        update = False

    if (stock['exchange'] == 'BSE' or stock['exchange'] == 'NSE'):
        tickertape = TickerTape(stock['tickertape_code'])
        df_financial = tickertape.get_yearly_finance()

        if stock_lastfinancial_year == None:
            df_financial = df_financial
        else:
            index = df_financial.index[df_financial['Financial_Year'] == stock_lastfinancial_year['Financial_Year']][0]
            #Last line of dataframe
            if df_financial.shape[0] == index + 1:
                print(f"{stock['StockName']} fundamental data is already upto date.")
                update = False
            else:
                df_financial = df_financial.tail(df_financial.shape[0] - (index+1))

    if update :
        for index, row in df_financial.iterrows():
            print(row.Financial_Year)
            query = f"""
            INSERT INTO stocksdb.raw_fundamental(stock_id,Financial_Year,Total_Revenue,EBITDA,PBIT,PBT,Net_Income,EPS,DPS,Payout_ratio)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """
            mycursor.execute(query,
                                (stock['ID'], row.Financial_Year, row.Total_Revenue,
                                row.EBITDA, row.PBIT, row.PBT, row.Net_Income,
                                row.EPS, row.DPS, row.Payout_ratio))

        conn.commit()
print("Done")
