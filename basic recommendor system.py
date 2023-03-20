#basrecrat, and basrecgen is the function.

import pandas as pd
import numpy as np
from ast import literal_eval  # evaluate strings containing Python code in the current Python environment
from nltk.stem.snowball import SnowballStemmer # Removing stem words
from sklearn.feature_extraction.text import CountVectorizer  # To convert text to numerical data
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.express as px#free graphing software in python
from wordcloud import WordCloud#some bakchodi for strings and stuff
import seaborn as sns#for plotting graphs underneath the matplotlib
import networkx as nx#for creating nodes and edges in the graph
import json
import re
import warnings  # disable python warnings
warnings.filterwarnings("ignore")
movies_data = pd.read_csv("movies_metadata.csv", low_memory=False)
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
links_small = pd.read_csv('links_small.csv')
ratings = pd.read_csv("ratings_small.csv")
movies_data = movies_data.dropna(subset=['vote_average', 'vote_count'])


def weighted_rating(v, R):
    '''

    This function calculate weighted rating of a movies using IMDB formula

    Parameters: v (int): vote count
                R (int): vote average
    Returns: (float) IMDB score

    '''
    return ((v / (v + m)) * R) + ((m / (m + v)) * C)


C = movies_data['vote_average'].mean()  # mean vote across all data
m = movies_data['vote_count'].quantile(0.95)  # movies with more than 95% votes is taken (95 percentile)

# Taking movies whose vote count is greater than m
top_movies = movies_data.copy().loc[movies_data['vote_count'] >= m]
top_movies = top_movies.reset_index()

top_movies['score'] = ''

for i in range(top_movies.shape[0]):
    v = top_movies['vote_count'][i]  # number of vote count of the movie
    R = top_movies['vote_average'][i]  # average rating of the movie
    top_movies['score'][i] = weighted_rating(v, R)

top_movies = top_movies.sort_values('score', ascending=False)  # sorting movies in descending order according to score
top_movies = top_movies.reset_index()

# top_movies[['title', 'vote_count', 'vote_average', 'score']].head(20) # top 20 movies
t1 = top_movies[['title', 'score','id']].head(20)
def basrecrat():
    for i in t1['id']:
        print(i)# put this inside a function to get recommendations based on the user rating



                                                    #genre based.
genres = set()

for i in range(top_movies['genres'].shape[0]):
    top_movies['genres'][i] = top_movies['genres'][i].replace("'", '"')

    # Ensure that all keys are enclosed in double quotes
    top_movies['genres'][i] = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', top_movies['genres'][i])

    # Parse the JSON string
    top_movies['genres'][i] = json.loads(top_movies['genres'][i])
    for x in top_movies['genres'][i]:
        genres.add(x['name'])

# creating map of string (genre name) and movies names(dataframe)
genres_based = dict()
for i in range(top_movies['genres'].shape[0]):
    for x in top_movies['genres'][i]:
        if x['name'] not in genres_based.keys():
            genres_based[x['name']] = pd.DataFrame(columns=top_movies.columns)
        genres_based[x['name']] = genres_based[x['name']].append(top_movies.iloc[i])

    # In[9]:

# Visualizing frequency of occurence of different genres

# Creating a count vector (list) containing frequency of a perticular genre
cnt = list()
for i in genres:
    cnt.append(genres_based[i].shape[0])

# Making a dataframe
genre_cnt = pd.DataFrame({'genres': list(genres),
                          'count': cnt

                          },
                         columns=['genres', 'count']
                         )

# fig = px.bar(genre_cnt, x='genres', y='count')
# fig.show()

# In[10]:


# print(top_movies)


# In[11]:


def genres_based_rcmnd(name):
    '''

    This function returns the top 10 movies of the given genre

    Parameters: name (string): Name of the genre

    Returns: (Dataframe) Top 10 move recommendation

    '''

    if name not in genres:
        return None
    else:
        return genres_based[name][['title', 'vote_count', 'vote_average', 'score','id']].head(10)


# print(genres_based_rcmnd("War"))
def basrecgen(name):
    for i in genres_based_rcmnd(name)['id']:
        print(i)#put this inside a function to get the recommendations based on the genres. for the final cut.
