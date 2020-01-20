#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
import pandas as pd
import random
import re, string
import pprint 
from collections import defaultdict

# nltk library classes
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

# Database imports to pull in Twitter data
from tt_jupyter.database import default_connection_context
from tt_jupyter.twitter import *
from cs_pg_common.pg.base import transaction_wrapper


# In[ ]:


nltk.download('twitter_samples') # We will use the positive, negative, and non-labeled tweets from this dataset (http://www.nltk.org/howto/corpus.html)
nltk.download('punkt')           # Will use .tokenize() from this class in order to tokenize a tweet 
nltk.download('stopwords')       # will use .words('English') in order to remove stop words (i.e. a, an, the) from tweet
nltk.download('wordnet')         # This will help determine the base word (i.e. running -> run)
nltk.download('averaged_perceptron_tagger') # This will determine the context of a word in a sentence to correctly tokenize 


# ### Function definitions

# In[ ]:


def clean_tweet(tokens):
    """Returns a cleaned tweet with new tag - a clean tweet includes removing any hyperlink and the '@'
       from the tweet, removing replacing the position tag with a n, v, or a so as to be able to lemmatize
       in the next step. Then finally remove any punctuation and return in lower case if the token is not 
       a stop word.
    """
    
    cleaned_tokens = []

    for token, tag in pos_tag(tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lem = WordNetLemmatizer()
        token = lem.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stopwords.words('english'):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# In[ ]:


def cleaned_tweets_dict(tokens_list):
    """Algorithm to return each token in the tokens list as a dictionary key:value pair
       with a value of true
    """
    for tokens in tokens_list:
        yield dict([token, True] for token in tokens)


# In[ ]:


def get_words(tokens_list):
    """Algorithm to return each token in the given list to place into another list"""
    for tokens in tokens_list:
        for token in tokens:
            yield token


# In[ ]:


def tweet_totals(candidate_name):
    """Returns the name, total number of positive tweets, total number of negative tweets, and
       sum of positive and negative tweets of each candidate name given into the function
       
       The following function is what I was provided with in order to search the third-party
       database I've been given access to. The function takes the following paramaters:
       
       search_tweets(connection, opt_content=None, opt_author_ids=None, opt_author_screen_names=None,
                  opt_tweet_identifiers=None, opt_in_reply_to_tweet_identifiers=None, opt_symbols=None,
                  opt_hashtags=None, conjunction='AND', limit=None, offset=None)
    """
    with default_connection_context() as conn:
        
        at_a_time = 1000
        search_string = '%'+ candidate_name + '%' # Need to include % in order to search
        current_offset = 0
        tweet_total = 0
        positive_total = 0
        negative_total = 0
        tweets = True
        
        while tweets:
            tweets = search_tweets(conn, opt_content=search_string, conjunction='AND', 
                                   limit=at_a_time, offset=current_offset)
            tweet_total += len(tweets)
            for tweet in tweets:
                custom_token = clean_tweet(word_tokenize(tweet.tweet_text))
                result = classifier.classify(dict([token, True] for token in custom_token))
                if result == 'Positive':
                    positive_total += 1
                    pass
                elif result == 'Negative':
                    negative_total += 1
                    pass
                else:
                    assert False, result
            current_offset += at_a_time
            
        total_tweets = positive_total + negative_total
                
    return candidate_name, positive_total, negative_total, total_tweets


# In[ ]:


candidate_name = ("(B|b)ernie (S|s)anders", "(A|a)ndrew (Y|y)ang", "(J|j)oe (B|b)iden", "(C|c)ory (B|b)ooker", 
                  "(P|p)ete (B|b)uttigieg", "(A|a)my (K|k)lobuchar", "(T|t)ulsi (G|g)abbard", 
                  "(E|e)lizabeth (W|w)arren", "(M|m)arianne (W|w)illiamson", 
                  "(J|j)ulian (C|c)astro", "(T|t)om (S|s)teyer", "(M|m)ike (B|b)loomberg")

data_list = []

for name in candidate_name: 
    data_list.append(list(tweet_totals(name)))
    
cols = ['Candidate', 'Positive Tweets', 'Negative Tweets', 'Total']
df = pd.DataFrame(data_list, columns = cols)

df


# In[ ]:


# Code used to clean up data frame above.
df_copy = df
proper_names = pd.DataFrame({'Candidate':["Bernie Sanders","Andrew Yang", "Joe Biden", "Cory Booker", 
                  "Pete Buttigieg", "Amy Klobuchar", "Tulsi Gabbard", 
                  "Elizabeth Warren", "Marianne Williamson", 
                  "Julian Castro", "Tom Steyer", "Mike Bloomberg"]})

df_copy = df_copy.drop(['Candidate'], axis = 1)
df_copy = df_copy.join(proper_names)
df_copy = df_copy[cols]

sum_total_positive = df_copy['Positive Tweets'].sum()
sum_total_negative = df_copy['Negative Tweets'].sum()
sum_total = df_copy['Total'].sum()

df_copy['Percentage Positive'] = (df_copy['Positive Tweets']/sum_total_positive)*100
df_copy['Percentage Negative'] = (df_copy['Negative Tweets']/sum_total_negative)*100
df_copy['Percentage Total'] = (df_copy['Total']/sum_total)*100

df_copy = df_copy.sort_values(by = 'Percentage Positive', ascending = False)
df_copy = df_copy.reset_index(drop=True)
df_copy

df_total_positive = df_copy[['Candidate', 'Percentage Positive']]
df_total_positive.set_index('Candidate', inplace = True)
df_total_positive = df_total_positive.sort_values(by = 'Candidate')
df_total_positive


# In[ ]:


# Data pulled from reference [4], dataframe cleaned up
poll_candidate = ["Joe Biden", "Bernie Sanders", "Elizabeth Warren", "Pete Buttigieg",
                        "Mike Bloomberg","Andrew Yang","Amy Klobuchar","Cory Booker","Tulsi Gabbard",
                        "Tom Steyer","Julian Castro", "Marianne Williamson"]
RCP_Average = [27.0, 18.6, 15.9, 9.1, 5.4, 3.1, 3.1, 2.8, 1.5, 1.3, 0.9, 0.3]
poll_df = pd.DataFrame({"Candidate":poll_candidate})
poll_df['Poll Percentage'] = RCP_Average 
poll_df

poll_df_copy = poll_df.sort_values(by = 'Candidate')
candidate_list = list(poll_df_copy['Candidate'])
poll_df_copy = poll_df_copy.set_index('Candidate')
poll_df_copy


# In[ ]:


# Bar chart plot code
import numpy as np
import matplotlib.pyplot as plt

n_groups = 12
fig, ax = plt.subplots(figsize = (18,10))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, df_total_positive['Percentage Positive'], bar_width,
alpha=opacity,
color='b',
label='Twitter Data')

rects2 = plt.bar(index + bar_width, poll_df_copy['Poll Percentage'], bar_width,
alpha=opacity,
color='g',
label='Poll Data')

plt.xlabel('Candidate')
plt.ylabel('Percentage')
plt.title('Percentages by Candidate')
plt.xticks(index + bar_width, candidate_list)
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


# Pie graph chart code:
df_total_positive.plot.pie( y = 'Percentage Positive', figsize = (20,20))
poll_df_copy.plot.pie( y = 'Poll Percentage', figsize = (20,20))

