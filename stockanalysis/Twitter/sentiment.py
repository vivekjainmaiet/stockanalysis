from textblob import TextBlob

def string_to_sentiment(text):

    return TextBlob(text).sentiment.polarity
