from stockanalysis.data import *
from stockanalysis.utils import *
from stockanalysis.param import *
import joblib
import os
import datetime

import mysql.connector as connection
from param import config

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from google.cloud import storage

PATH_TO_LOCAL_MODEL = 'model.joblib'
BUCKET_NAME = "one-stop-stock-analysis"


def split_timeseries(y, X, sequence_size=21):
    # Create the training data set
    # Create the scaled training data set
    train_data = y
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(sequence_size, len(train_data)):

        x_train.append(train_data[i - sequence_size:i, :])
        y_train.append(X[i])

        if i <= sequence_size+1:
            # print(x_train)
            # print(y_train)
            print('i <= sequence_size+1')

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    return x_train, y_train


def set_pipeline(cleaned_data):
    '''returns a pipelined model'''
    data_pipe = Pipeline([('stdscaler', MinMaxScaler())])
    preproc_pipe = ColumnTransformer(
        [('data', data_pipe,
          make_column_selector(dtype_include=["int64", "float64"]))],
        remainder="drop")

    pipe = Pipeline([('preproc', preproc_pipe)])

    scaled_data = pipe.fit_transform(cleaned_data)
    return scaled_data, pipe


def download_model(storage_location='models/stockanalysis/Pipeline/INFY.NS.joblib',bucket=BUCKET_NAME,rm=False):
    client = storage.Client().bucket(bucket)
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model


def prediction(ticker, start, end):
    #Importing Information
    online = False
    print(ticker, start, end)
    df = get_technical(symbol=ticker, start=start, end=end)
    print(df.tail(5))
    cleaned_data = clean_data(df.drop(columns=['Date']))[COLUMNS]

    #Creating Sequence to pass to the prediction
    X = cleaned_data.to_numpy()[-(SEQUENCE_SIZE+1):, :]
    y = cleaned_data['Close'].to_numpy()[-(SEQUENCE_SIZE+1):]
    X, y = split_timeseries(X, y, sequence_size=SEQUENCE_SIZE)

    #Load model trainned model in previous stage to predict future price
    if online == True:
        model = download_model(storage_location=f'models/stockanalysis/Pipeline/{ticker}.joblib',bucket=BUCKET_NAME,rm=False)
    else:
        model = joblib.load(f'{ticker}.joblib')


    #Model results
    results = model.predict(X)

    pred_values = np.exp(results[0])
    pct_change_predictions = pd.Series(np.exp(results[0])).pct_change() * 100
    cumsum_pctchange_predictions = pct_change_predictions.cumsum()
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)


    return {
        "prediction": pred_values,
        'actual_prices': y,
        'pct_change_predictions': pct_change_predictions,
        'cumsum_pctchange_predictions': cumsum_pctchange_predictions,
        'model': {
            "model": model.to_json(),
            "COLUMNS": COLUMNS,
            "SEQUENCE_SIZE": SEQUENCE_SIZE
        }
    }


if __name__ == "__main__":
    import datetime
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    ticker = "AAPL"
    print(start_date, end_date)
    pred = prediction(ticker=ticker, start=start_date, end=end_date)
    df_pred = pd.DataFrame(pred['prediction'])
    df_pct_change_predictions = pd.DataFrame(pred['pct_change_predictions'])
    df_cumsum_pctchange_predictions = pd.DataFrame(pred['cumsum_pctchange_predictions'])
    df_model_summary = pd.DataFrame(pred['model'])

    conn = connection.connect(**config)
    mycursor = conn.cursor(dictionary=True)
    query = f"SELECT * FROM stocksdb.StocksList where yahoo_code ='{ticker}';"
    mycursor.execute(query)
    stock = mycursor.fetchone()
    stock_id = stock['ID']


    df_pred = np.array2string(df_pred.T.values,
                           formatter={'float_kind': lambda x: "%.2f" % x})
    df_pct_change_predictions = np.array2string(
        df_pct_change_predictions.T.values,
        formatter={'float_kind': lambda x: "%.2f" % x})
    df_cumsum_pctchange_predictions = np.array2string(
        df_cumsum_pctchange_predictions.T.values,
        formatter={'float_kind': lambda x: "%.2f" % x})
    x = datetime.datetime.now()

    query = f"""
            INSERT INTO stocksdb.Stock_Prediction(stock_id,date,model,prediction_price,prediction_perchange,prediction_cum_perchange)
            VALUES (%s,%s,%s,%s,%s,%s);
            """
    mycursor.execute(
        query, (stock_id, x, df_model_summary.to_string(), df_pred,
                df_pct_change_predictions,
                df_cumsum_pctchange_predictions))

    conn.commit()

    #breakpoint()
