import tweepy

# Creating an API object
# Creating an API object

SEQUENCE_SIZE = 50
Y_LEN = 7
#pct_change, log, normal, diff
TYPE_Y = 'log'
PATIENCE = 50
# COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'ema12', 'ema21',
#        'ema26', 'ema34', 'ema55', 'ema99', 'ema200', 'hma12', 'hma21', 'hma26',
#        'hma34', 'hma55', 'hma99', 'hma200', 'rsi', 'atr', 'bb_upper',
#        'bb_lower', 'macd_signal', 'macd_line', 'adx', 'vwap']
# COLUMNS = ['Volume', 'rsi', 'macd_line', 'ema21', 'bb_upper', 'bb_lower', 'ema200','High', 'Low', 'Close']
COLUMNS = ['Volume', 'rsi','Close']

config = {
    'user': 'users',
    'password': '#Stocks@007#',
    'host': '34.79.163.70',
    'database': 'stocksdb'
}
client = tweepy.Client(
    bearer_token=
    'AAAAAAAAAAAAAAAAAAAAAD3fYgEAAAAAcMN%2BYuat3oqoeE%2BoADUe9ZPwbQM%3Dcj3qBljHJs5WVHHqzANEYp9fpOQjgVZIr0fV4stVQDAYrHinXA'
)

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'one-stop-stock-analysis'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'stockanalysis'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'Pipeline'
