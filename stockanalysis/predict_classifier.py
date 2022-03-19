import pandas as pd
import numpy as np
import joblib
from datetime import datetime,timedelta
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import yfinance as yf
yf.pdr_override()

import mysql.connector as connection
from param import config


def predict(ticker="TCS.BO",path="/Users/vivek/code/vivekjainmaiet/stockanalysis/Reverse.joblib"):

    model = joblib.load(path)

    N_DAYS_AGO = 50

    today = datetime.now()
    n_days_ago = today - timedelta(days=N_DAYS_AGO)

    start = n_days_ago.date()
    end = today.date()

    # download dataframe
    df = pdr.get_data_yahoo(ticker, start=start, end=end).reset_index()

    df['change'] = (((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) *100).round(2).dropna()
    new_df = df['change']

    data_set_points = 21

    test_data = new_df[-data_set_points - 2:]
    test_data = test_data.reset_index()
    test_data = test_data.drop(columns=['index'])

    test_arr = np.diff(test_data.loc[:, :].values, axis=0)
    X_test = np.array([test_arr[i:i + data_set_points]for i in range(len(test_arr) - data_set_points)])

    result = model.predict(X_test)
    print("Predicted percentage change is:", result)

    if result < -1.5:
        Action = "BUY"
    elif result > 1.5:
        Action = "SELL"
    else:
        Action = "No Recommendation"

    print(f"{ticker} Model recommendation for next trading day is {Action}")

    conn = connection.connect(**config)
    mycursor = conn.cursor(dictionary=True)
    query = f"SELECT * FROM stocksdb.StocksList where yahoo_code ='{ticker}';"
    mycursor.execute(query)
    stock = mycursor.fetchone()
    stock_id = stock['ID']

    x = datetime.now()

    query = f"""
            INSERT INTO stocksdb.Stock_Prediction(stock_id,date,model,action)
            VALUES (%s,%s,%s,%s);
            """
    mycursor.execute(query, (stock_id, x, "LSTM Reverse Classifier", Action))

    conn.commit()



if __name__ == "__main__":
    predict(ticker="TCS.BO")
    predict(ticker="INFY.BO")
    predict(ticker="AAPL")
