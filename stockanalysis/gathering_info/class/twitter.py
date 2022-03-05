import tweepy
import pandas as pd
import numpy as np
from stockanalysis.utils import *
from stockanalysis.param import *
from datetime import date, datetime, timedelta


class Scraper:
    def __init__(self, stock_name, max_results):
        self.stock_name = stock_name
        self.max_results = max_results

        today = date.today()
        d1 = str(today.strftime("%Y-%m-%d")) + 'T00:00:00Z'

        d2 = datetime.now() - timedelta(days=1)

        self.today = d1
        self.yesterday = str(d2.strftime("%Y-%m-%d")) + 'T00:00:00Z'
        self.dataframe = None  #dataframe to update

    def get_tweets(self):
        output = []  #list of tweets
        text_query = f'{self.stock_name} -is:retweet lang:en'  #no retweets

        ### Creation of query method using parameters###
        for tweet in tweepy.Paginator(client.search_recent_tweets, query=text_query,tweet_fields=['text', 'created_at'],\
                                      start_time= self.yesterday, end_time=self.today, \
                                      max_results=self.max_results).flatten(limit=10): #to use more than 100 tweets

            #for tweet in tweets_list.data:
            text = tweet.text
            created_at = tweet.created_at
            line = {'text': text, 'created_at': created_at}
            output.append(line)

        df = pd.DataFrame(output)
        self.dataframe = df  #UPDATE self.dataframe

    def preprocess_tweets(self):
        '''preprocess and clean text of all tweets'''
        df = self.dataframe
        # remove twitter handles (@user)
        df['clean_text'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")
        # remove special characters, numbers, punctuations
        df['clean_text'] = df['clean_text'].str.replace("[^a-zA-Z#]", " ")
        #removing short words
        df['clean_text'] = df['clean_text'].apply(
            lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
        #lowercase text
        df['clean_text'] = df['clean_text'].apply(lower)

    def create_sentiment(self):
        '''Create sentiment from Tweets --> output range [-1,1]'''
        df = self.dataframe
        df['sentiment'] = df['clean_text'].apply(string_to_sentiment)
        df = df.round(2)  #round numbers to 2 decimals
        return df


if __name__ == "__main__":

    scraper = Scraper('Bitcoin', 10)
    scraper.get_tweets()
    scraper.preprocess_tweets()
    df = scraper.create_sentiment()
    print(df)
