# $DELETE_BEGIN
import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import mysql.connector as connection
from stockanalysis.param import config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return dict(stocklist="/stocklist",
                stock="/stock?ticker=TCS",
                technical="/technical?ticker=TCS",
                news="/newslist?ticker=TCS",
                recommendation="/recommendation?ticker=TCS")


@app.get("/stocklist")
def stocklist():  # 1
    conn = connection.connect(**config)
    mycursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM stocksdb.StocksList;"
    mycursor.execute(query)
    stock_list = mycursor.fetchall()
    return stock_list


@app.get("/stock")
def stock(ticker):  # 1
    conn = connection.connect(**config)
    mycursor = conn.cursor(dictionary=True)
    query = f"SELECT * FROM stocksdb.StocksList where StockCode ='{ticker}';"
    mycursor.execute(query)
    stock = mycursor.fetchone()
    return stock

@app.get("/technical")
def technical(ticker):  # 1
    conn = connection.connect(**config)
    mycursor = conn.cursor(dictionary=True)
    query = f"SELECT * FROM stocksdb.StocksList where StockCode ='{ticker}';"
    mycursor.execute(query)
    stock = mycursor.fetchone()
    stock_id = stock['ID']
    query = f"SELECT * FROM stocksdb.raw_technical where Stock_id = {stock_id}"
    mycursor.execute(query)
    technical_data = mycursor.fetchall()
    return technical_data

@app.get("/newslist")
def newslist(ticker):
    conn = connection.connect(**config)
    mycursor = conn.cursor(dictionary=True)
    query = f"SELECT * FROM stocksdb.StocksList where StockCode ='{ticker}';"
    mycursor.execute(query)
    stock = mycursor.fetchone()
    stock_id = stock['ID']
    query = f"SELECT * FROM stocksdb.raw_news where stock_id = {stock_id};"
    mycursor.execute(query)
    newslist = mycursor.fetchall()
    return newslist


@app.get("/recommendation")
def recommendation(ticker):  # 1
    conn = connection.connect(**config)
    mycursor = conn.cursor(dictionary=True)
    query = f"SELECT * FROM stocksdb.StocksList where StockCode ='{ticker}';"
    mycursor.execute(query)
    stock = mycursor.fetchone()
    stock_id = stock['ID']
    query = f"SELECT * FROM stocksdb.raw_recommendation where stock_id = {stock_id};"
    mycursor.execute(query)
    recommendation_list = mycursor.fetchall()
    return recommendation_list


@app.get("/prediction")
def prediction(ticker):
    #Load model trainned model in previous stage to predict future price
    model = joblib.load('/content/drive/MyDrive/Colab Notebooks/02_24_2022_20_37_30_model.joblib')
    results = model.predict()
    pred = float(results[0])
    return pred
