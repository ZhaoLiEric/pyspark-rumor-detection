import json
import socket

import tweepy
from tweepy import StreamListener, Stream

tweets = []

class TweetsListener(StreamListener):
    def __init__(self, socket: socket.socket):
        self.client_socket = socket
        super().__init__()

    def on_data(self, raw_data):
        try:
            self.client_socket.send(raw_data.encode('utf-8'))

        except BaseException as e:
            print(f"error on_data: {e}")
        return True

    def on_error(self, status_code):
        print(f"error on_error: {status_code}")
        return True


consumer_key = ""
consumer_secret = ""
bearer_token = ""


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token("","")


host = "localhost"
port = 9999
tracks = "#SuperLeague".split()
#
s = socket.socket()
s.bind((host, port))
s.listen(5)

connection, client_address = s.accept()
twitter_stream = Stream(auth, TweetsListener(connection))
twitter_stream.filter(track=tracks, languages=['en'])


#