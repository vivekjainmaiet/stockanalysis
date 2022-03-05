import tweepy
from stockanalysis.param import *
from stockanalysis.utils import *
from stockanalysis.gathering_info.classes.sentiment import *
import pandas as pd
import numpy as np
import datetime

#-> (PREDICT)
# MONDAY->TUESDAY->WEDNESDAY->THURSDAY->FRIDAY->MONDAY...


class Scraper:
    def __init__(self, stock_name, max_results):
        self.stock_name = stock_name
        self.max_results = max_results

        #GLOBAL CLASS ATTRIBUTES#

        self.today = str(datetime.date.today()
                         ) + 'T00:00:00Z'  #I need a  RFC 3339 timestamp format
        self.yesterday = str(datetime.date.today() -
                             datetime.timedelta(days=1)) + 'T00:00:00Z'
        self.dataframe = None  #dataframe to update

#___________________________________________________________________________________________________________________________________________

    def get_tweets(self):
        '''Download tweets and store them into a Dataframe:

        stock_name => ('AAPL', 'AMZN' etc)
        TCS TATA CONSULTANCY SERVICES
        since_date =>  '2022-02-09' e.g.
        until_date = '2022-02-10' e.g. #to calculate one specific day, set until-date as the next day
        max_tweets => limit of tweets to scrape'''

        output = []  #list of tweets
        query = ' OR '.join(self.stock_name.split(","))
        text_query = f'{query} -is:retweet lang:en'  #no retweets

        ### Creation of query method using parameters###

        for tweet in tweepy.Paginator(client.search_recent_tweets, query=text_query,tweet_fields=['text', 'created_at'],\
                                      start_time= self.yesterday, end_time=self.today, \
                                      max_results=100).flatten(limit=self.max_results): #to use more than 100 tweets (max 2.000.000 per month)

            #for tweet in tweets_list.data:

            text = tweet.text
            created_at = tweet.created_at
            line = {'text': text, 'created_at': created_at}
            output.append(line)

        df = pd.DataFrame(output)

        self.dataframe = df  #UPDATE self.dataframe

#________________________________________________________________________________________________________________________________________

    def preprocess_tweets(self):
        '''preprocess and clean text of all tweets'''

        df = self.dataframe

        #lowercase text
        df['clean_text'] = df['text'].apply(lower)

        # remove numbers,punctuation,spaces
        df['clean_text'] = df['clean_text'].apply(clean_twitter_text)

        #remove stock name
        df['clean_text'] = df['clean_text'].str.replace(
            self.stock_name.lower(), "")


        #drop spam tweets (containing http or https words)
        #df = df[df["clean_text"].str.contains("https|http") == False]

#_________________________________________________________________________________________________________________________________________

    def create_sentiment(self):
        '''Create sentiment from Tweets --> output range [-1,0,1]'''

        df = self.dataframe

        df['sentiment'] = df['clean_text'].apply(custom_sentiment_base)


#_________________________________________________________________________________________________________________________________________

    def save_df(self):
        '''save 2 dataframes'''

        #save df
        df_final = self.dataframe

        date = df_final['created_at'][0].strftime(
            '%Y-%m-%d')  #convert timestamp to str
        #file_name = f'{date}.csv'
        #SAVE COMPLETE DF WITH TWEETS
        #df_final.to_csv(f'/Users/vivek/code/vivekjainmaiet/stockanalysis/raw_data/{self.stock_name}/{file_name}',index=False)

        #SAVE MAIN DF WITH ONLY NECESSARY DATA

        pos = df_final.sentiment[df_final.sentiment == 'pos'].count()
        neg = df_final.sentiment[df_final.sentiment == 'neg'].count()
        df_final = pd.DataFrame([{'Date': date, 'pos': pos, 'neg': neg}])
        #breakpoint()
        return df_final

        #file_name_2= f'report_{date}_{self.stock_name}.csv'
        #df_final.to_csv(f'/Users/vivek/code/vivekjainmaiet/stockanalysis/raw_data/{file_name_2}',index=False)

    #def write_sentiment_df(self):

    #    df= self.dataframe
    #    pos = df.sentiment[df.sentiment == 'pos'].count()
    #    neg = df.sentiment[df.sentiment == 'neg'].count()

    #    # open the file in the write mode
    #    with open(
    #            '/home/lorisliusso/code/lorisliusso/twitter_project/Twitter/data/reports/report_AAPL.csv',
    #            'w') as f:
    #        # create the csv writer
    #        writer = csv.writer(f)
    #        # write a row to the csv file
    #        date = df['created_at'][0].strftime('%Y-%m-%d')
    #        row= f'{date},{pos},{neg}'
    #        writer.writerow(row)


#________________________________________________________________________________________________________________________________________

if __name__ == "__main__":

    scraper = Scraper('TCS', 10)
    scraper.get_tweets()
    scraper.preprocess_tweets()
    scraper.create_sentiment()
    df = scraper.save_df()
    print(df)
