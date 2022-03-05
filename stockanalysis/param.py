import tweepy

# Creating an API object
# Creating an API object
SEQUENCE_SIZE = 21
PATIENCE = 10
COLUMNS = ['Volume', 'rsi', 'macd_line', 'ema21', 'bb_upper', 'bb_lower', 'ema200','High', 'Low', 'Close']
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
