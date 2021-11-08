#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests
import bs4
from bs4 import BeautifulSoup
from requests_oauthlib import OAuth1
import pandas as pd
import tweepy
import time


# In[2]:


df  = pd.read_csv('upto_userName.csv')


# In[3]:


df


# In[4]:


consumer_key = '7dzK7q6IrWCcq5Kjo3q5YDrGJ'
consumer_secret = 'd52zgvIjSq53L4TIltmpbN4iE0vZtrZAVkdz0Fw2YHdlOqwMS7'
access_token = '948413893038125056-u5qBXf9dtb4aD4MJKewEIk75N7Tf40B'
access_token_secret= '6pdoGlxNDFgHvB3Y1tTJVTTD3UXddCSf4ABBZVMh1xPxC' 

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)


# In[5]:


tweetFetched = api.get_status(573596258201268224)


# In[6]:


print(tweetFetched.text)


# In[7]:


# df['tweet_text']=df.iloc[:50,0].apply(lambda x: tweet_text_from_tweet_id(x))


# In[8]:


df


# In[9]:


# Status Methods
def tweet_text_from_tweet_id(idx, API):
    return API.get_status(idx).text


# User Methods
def get_followers(screen_name,api):
    user_ids = []
    for page in tweepy.Cursor(api.followers_ids,
                              screen_name=screen_name).pages():
        user_ids.extend(page)
        time.sleep(60)

    return user_ids


def user_from_tweet_id(idx, api):
    status = api.get_status(idx)
    return (status.user.id_str)


def username_from_user_id(idx, api):
    user = api.get_user(user_id=idx)
    return user.screen_name


def timeline_from_username(screen_name, api):
    timeline = api.user_timeline(screen_name=screen_name)
    return timeline

def retweet_users_from_tweet_id(idx, api):
    # getting the retweeters
    retweets_list = api.retweets(idx)

    # printing the screen names of the retweeters
    retweet_users=[]
    for retweet in retweets_list:
        retweet_users.append(retweet.user.screen_name)
        
    return retweet_users


def get_follow_info(x, y, api):
    api.show_friendship(source_id=x, target_id=y)


# In[10]:


def def_call(def_name,x):
    try:
        return def_name(x,api)

    except:
        return np.nan


# In[10]:


df['tweet text'] = df['tweet id'].apply(lambda x: def_call(tweet_text_from_tweet_id,x))

temp = df.copy()
df.dropna(inplace=True)

df['tweet user id'] = df['tweet id'].apply(lambda x: def_call(user_from_tweet_id,x))


# In[55]:


temp = df[df['tweet user id'].isna()]['tweet id'].apply(lambda x: def_call(user_from_tweet_id,x))

a = list(df.iloc[:,:-1].dropna()['tweet user id'].values)
b = list(temp.values)
df['tweet user id'] = a+b


# In[60]:


df['tweet user'] = df['tweet user id'].apply(lambda x: def_call(username_from_user_id,x))

temp = df[df.isna().any(axis=1)]['tweet user id'].apply(lambda x: def_call(username_from_user_id,x))
a = list(df.dropna()['tweet user'].values)
b = list(temp.values)
df['tweet user'] = a+b


# In[118]:


#df.to_csv('upto_userName.csv',index=False)


# In[8]:


df.dropna(inplace=True)
def getFriendship(src,tar):
    followed_by=False
    following =False
    friendship = api.show_friendship(source_id = src, target_id = tar)
    if friendship[0].followed_by:
        followed_by = True
    if friendship[0].following:
        following = True
        
    return following, followed_by


# In[28]:


getFriendship(int(13857342.0),int(13857342.0))


# In[27]:


followed_by=[]
following =[]
for i in df['tweet user id'].unique():
    l = list(df['tweet user id'].unique())
    l.remove(i)
    f1=[]
    f2 =[]
    for j in l:
        res = getFriendship(int(i),int(j))
        print(res)
        if res[0]:
            f1.append(j)
        if res[1]:
            f2.append(j)
    print(f1,f2)  
    followed_by.append(f1)
    following.append(f2)


# In[11]:


df


# In[14]:


df['user followers'] = df['tweet user'].apply(lambda x: def_call(get_followers,x))


# In[13]:


df


# In[12]:


get_followers('YesYoureSexist',api)


# In[ ]:


# 1df['user followers'] = df['tweet user'].apply(lambda x: def_call(get_followers,x))
df['user timeline'] = df['tweet user'].apply(lambda x: def_call(timeline_from_username,x))

df['retweet user'] = df['tweet id'].apply(lambda x: def_call(retweet_users_from_tweet_id,x))
df['follow info'] = df['tweet id'].apply(lambda x: get_follow_info(x))


df.to_csv('tweet_abuse_data.csv',index=False)


# In[ ]:


# tweet_id, tweet_content, user_id, user_name, label, retweet_list


# In[ ]:





# In[ ]:




