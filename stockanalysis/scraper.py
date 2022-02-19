import tweepy
import datetime as datetime
import pandas as pd
import numpy as np
from textblob import TextBlob
from param import *
from utils import *

#-> (PREDICT)
# MONDAY->TUESDAY->WEDNESDAY->THURSDAY->FRIDAY->MONDAY...


class Scraper:

    def __init__(self, stock_name, max_results):
        self.stock_name = stock_name
        self.max_results= max_results

        #GLOBAL CLASS ATTRIBUTES#

        self.today = str(datetime.date.today())+'T00:00:00Z'  #I need a  RFC 3339 timestamp format
        self.yesterday = str(datetime.date.today() - datetime.timedelta(days=1)) + 'T00:00:00Z'
        self.dataframe= None #dataframe to update

#___________________________________________________________________________________________________________________________________________

    def get_tweets(self):

        '''Download tweets and store them into a Dataframe:

        stock_name => ('AAPL', 'AMZN' etc)
        since_date =>  '2022-02-09' e.g.
        until_date = '2022-02-10' e.g. #to calculate one specific day, set until-date as the next day
        max_tweets => limit of tweets to scrape'''

        output = [] #list of tweets
        text_query = f'{self.stock_name} -is:retweet lang:en'  #no retweets

        ### Creation of query method using parameters###

        #tweets_list= client.search_recent_tweets(query=text_query, tweet_fields=['text', 'created_at'], \
        #                                         start_time= self.yesterday, end_time=self.today, max_results=self.max_results)

        for tweet in tweepy.Paginator(client.search_recent_tweets, query=text_query,tweet_fields=['text', 'created_at'],\
                                      start_time= self.yesterday, end_time=self.today, \
                                      max_results=self.max_results).flatten(limit=200): #to use more than 100 tweets

            #for tweet in tweets_list.data:

            text = tweet.text
            created_at = tweet.created_at
            line = {'text' : text, 'created_at' : created_at}
            output.append(line)

        df = pd.DataFrame(output)

        self.dataframe= df #UPDATE self.dataframe

#________________________________________________________________________________________________________________________________________

    def preprocess_tweets(self):

        '''preprocess and clean text of all tweets'''

        df= self.dataframe

        # remove twitter handles (@user)
        df['clean_text'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")

        # remove special characters, numbers, punctuations
        df['clean_text'] = df['clean_text'].str.replace("[^a-zA-Z#]", " ")

        #removing short words
        df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#_________________________________________________________________________________________________________________________________________

    def create_sentiment(self):

        '''Create sentiment from Tweets --> output range [-1,1]'''

        df= self.dataframe

        df['sentiment']= df['clean_text'].apply(string_to_sentiment)

        df=df.round(2) #round numbers to 2 decimals

#_________________________________________________________________________________________________________________________________________

    def save_df(self):

        '''save dataframe'''

        #save df
        df_final=self.dataframe

        date= df_final['created_at'][0].strftime('%Y-%m-%d') #convert timestamp to str
        file_name= f'{date}.csv'
        #df_final.to_csv(f'/Users/vivek/code/vivekjainmaiet/stockanalysis/{file_name}')
        return df_final

#________________________________________________________________________________________________________________________________________

if __name__ == "__main__":

    scraper = Scraper('apple Apple AAPL', 100)
    scraper.get_tweets()
    scraper.preprocess_tweets()
    scraper.create_sentiment()
    scraper.save_df()
