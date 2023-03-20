#colfil is the function

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
# ## Collaborative filtering
#
# Collaborative Filtering is based on the idea that users similar to a me can be used to predict how much I will like a particular product or service those users have used/experienced but I have not. We will be using Surprise library which uses algorithms like Singular Value Decomposition (SVD) to minimise RMSE (Root Mean Square Error) and other metrics and give great recommendations.
#
# Surprise is a Python scikit for building and analyzing recommender systems that deal with explicit rating data. It is taking as input ratings of few movies done by a set of users and predicting the rating of remaining movies by using collaborative filtering algorithm ( in this we are using SVD algorithm ).
#
# SVD creates a lower dimensional representation for each user and movie based on some latent factors(featurs of movies) and based on these lower dimensional representation model predicts rating of user for other movie. The dimension of latent factor can be controlled by n_factor parameter of a model ( its default value is 100 ). Usually, the quality of the training set predictions grows with as n_factors gets higher.
#
#
# Steps involved:
# 1. Preprocessing of data ( removing irrelevant columns, checking NAN values etc )
# 2. Using surprise library to make train and test dataset
# 3. Trained the model on the available data ( training dataset )
# 4. Finally, tested the quality of model by using RMSE and MAE
# 5. Now that our model is fitted, we can call predict to get some predictions. predict returns an internal object Prediction which can be easily converted back to a dataframe.
#
# Mean Absolute Error (MAE) measures the average magnitude of the errors in a set of predictions, without considering their direction.
# Root mean squared error (RMSE) is the square root of the average of squared differences between prediction and actual observation.
# Lower value of both RMSE and MAE is considered to be good.
#

# In[24]:
ratings = ratings.drop('timestamp', axis=1)

# checking for missing values
ratings.isna().sum()

# check for the numbers of total movies and users
movies = ratings['movieId'].nunique()  # nunique is similar to count but only takes unique values
users = ratings['userId'].nunique()
print('total number of movies =', movies)
print('total number of users =', users)

#  HIstogram showing frequency of ratings given by different users
# fig = px.histogram(ratings, x="rating")
# fig.show()


# In[30]:


# columns to use for training
columns = ['userId', 'movieId', 'rating']

# create reader from surprise
# the rating should lie in the provided scale
reader = Reader(rating_scale=(0.5, 5))

# create dataset from dataframe
data = Dataset.load_from_df(ratings[columns], reader)

# create trainset ie the data which is present (ratings of those movies which are rated by respective users)
trainset = data.build_full_trainset()

# create testset, here the anti_testset is testset
# data containing users movie pairs which are not rated by that particular user
testset = trainset.build_anti_testset()

model = SVD(n_epochs=25,
            verbose=False)  # n_epochs:The number of iteration of the SGD(simple gradient descent) procedure. Default is 20
# verbose:If True, prints the current epoch. Default is False.

cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
# print('Training Done')

# prediction
prediction = model.test(testset)

# ### Using user rating of selected movies to get suitable recommendations
#
# 1. To incorporate the user in the system we inspected the dataset and found that mostly users have around 5-10 ratings.
# 2. To get recommendation according to taste of user we prompt the user to rate the selected movies.
# 3. We add those selection to our standerd dataset and run the collaborative filtering algorithm to get the ratings of other movies.
# 4. Finally, recommending top 10 movies based on highest ratings.

# In[35]:


example = {'userId': [99999, 99999, 99999, 99999, 99999],
           'movieId': [31, 1029, 1293, 1172, 1343],
           'rating': [3.0, 4.5, 1.2, 3.3, 2]

           }

df = pd.DataFrame(example)
frames = [ratings, df]
result = pd.concat(frames)

# In[36]:


# create dataset from dataframe
data = Dataset.load_from_df(result[columns], reader)

# create trainset
trainset = data.build_full_trainset()

# create testset, here the anti_testset is testset
testset = trainset.build_anti_testset()

cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5,
               verbose=False)  # cv is the number of parts in which data will be divided.
# print('Training Done')

# prediction
prediction = model.test(testset)


# prediction[99999]


# In[37]:


def get_top_n(prediction, n):
    '''
    This function recommend users with top n movies based on prediction calculated using the surprise library

    Parameters: prediction(list): This contains (user, movie) rating prediction for all user movie pairs
                n(int): Number of recommendations

    Results: Returns top 30 movies along with movie id for all users


    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in prediction:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


rcmnd = []
top_n = get_top_n(prediction, n=30)
for uid, user_ratings in top_n.items():
    if uid == 99999:
        for (iid, rating) in user_ratings:
            for i in range(movies_data.shape[0]):
                if movies_data['id'][i] == iid:
                    rcmnd.append([movies_data['id'][i], movies_data['title'][i]])
        break

# In[40]:

def colfil():
    for i in rcmnd[:6]:
        print(i[0])


