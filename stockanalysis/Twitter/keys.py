import tweepy

consumer_key = "nzYbODEScixQ7p6aWiHdRxfmh"
consumer_secret = "oOhJV67RvwQQmYQM4KZ9DZiUiGdmQDJh4NQeOQKcohFwduMiQH"
access_key = "1488143650215374849-B2RhvHE3bG5uvlyt0n07ufkw5qgaV7"
access_secret = "SoH4CbFrdWUkRY7DSkhHGqtRkNTGp9Japl2FOTIyMI3kR"

# Twitter authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

# Creating an API object
# Creating an API object
client = tweepy.Client(
    bearer_token=
    'AAAAAAAAAAAAAAAAAAAAAD3fYgEAAAAAcMN%2BYuat3oqoeE%2BoADUe9ZPwbQM%3Dcj3qBljHJs5WVHHqzANEYp9fpOQjgVZIr0fV4stVQDAYrHinXA'
)
