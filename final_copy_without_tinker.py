#!/usr/bin/env python
# coding: utf-8

# # Guide for initial setup for the project
# 
# ### Install Anaconda in your machine to use Python and Jupyter Notebook
#        Reference:
#        For windows - https://docs.anaconda.com/anaconda/install/windows/
#        For Ubuntu  - https://docs.anaconda.com/anaconda/install/linux/
#        For MacOS   - https://docs.anaconda.com/anaconda/install/mac-os/
#    
# ### Create a local repository ( folder ) for the project 
#        2.1 Download all the dataset provided in the problem statement
#        2.2 Move all the dataset in project folder
# 
# ### To use Jupyter Notebook in Anaconda
#        3.1 Open "Conda" terminal in your machine
#        3.2 Navigate to project folder created in step-2
#        ( Ref - https://docs.anaconda.com/ae-notebooks/4.3.1/user-guide/basic-tasks/apps/use-terminal/)
#        3.3 Use the "jupyter notebook" command to open the jupyter notebook interface
#        ( Ref - https://docs.anaconda.com/ae-notebooks/user-guide/basic-tasks/apps/jupyter/index.html)
# 
# ### Installing the required python libraries
#        4.1 Download the provided requirements.txt file and move it to the project folder
#        4.2 Use the "pip install -r requirements.txt" command to install all of the required modules and packages.
#            This will install most of the required libraries for the project.
#            ( Ref - https://learnpython.com/blog/python-requirements-file/)
#        4.3 To install any other libraries refer
#            https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/
# 

# In[1]:


# Importing liberaries

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


# In[2]:


# Loading datasets 

movies_data = pd.read_csv("movies_metadata.csv", low_memory=False)
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
links_small = pd.read_csv('links_small.csv')
ratings = pd.read_csv("ratings_small.csv")


# In[3]:


movies_data.describe()


# In[4]:


# Checking for null values in the dataset

print(movies_data.isnull().sum(),'\n') # We have used only selected column of the dataset which include genres,id,vote_average,vote_count
print(links_small.isnull().sum(),'\n')
print(ratings.isnull().sum(),'\n')
print(keywords.isnull().sum(),'\n')
print(credits.isnull().sum(),'\n')


# In[3]:


# Removing the rows with null value in the vote_average and vote_count columns in movies_data dataframe

movies_data = movies_data.dropna(subset=['vote_average', 'vote_count'])
print(movies_data.isnull().sum(),'\n')


# ## SIMPLE RECOMMENDER SYSTEM
# 
# A simple recommender is a genral recommender system. It gives same recommendation to all the users irrespective of users preferences.The basic idea behing this recommender system is that a movies that is more popular will have the higher probability of being like by the average people.
# 
# Steps involved:
# 1. We will use IMDB's weighted rating formula to calculate the overall rating of a perticular movie and store it in column name 'score'.
# 
#     weighted rating: (v/v+m)*R + (m/m+v)*C
#     
#     where,
#            m = Minimum vote count required to be listed in chart.
#            v = Total number of votes of the movie (given in the dataset with column name 'vote_count')
#            R = Average rating of the movie (given in the dataset with column name 'vote_average' )
#            C = Average vote across all dataset (total vote divided by total movies)
#            
# 2. Then will sort the movie is accending order accoding to the score and finally will get the top movies.

# In[6]:


# Simple Recommender (Top movies irrespective of genres)

# Weighted rating
def weighted_rating(v,R):
    
    '''
    
    This function calculate weighted rating of a movies using IMDB formula
    
    Parameters: v (int): vote count
                R (int): vote average
    Returns: (float) IMDB score
    
    '''
    return ((v/(v+m)) * R) + ((m/(m+v)) * C)  



C = movies_data['vote_average'].mean()         # mean vote across all data
m = movies_data['vote_count'].quantile(0.95)   # movies with more than 95% votes is taken (95 percentile)

# Taking movies whose vote count is greater than m
top_movies = movies_data.copy().loc[movies_data['vote_count'] >= m]
top_movies = top_movies.reset_index()

top_movies['score'] = ''

for i in range(top_movies.shape[0]):
    v = top_movies['vote_count'][i]          # number of vote count of the movie
    R = top_movies['vote_average'][i]        # average rating of the movie
    top_movies['score'][i] = weighted_rating(v,R)

top_movies = top_movies.sort_values('score', ascending=False)  # sorting movies in descending order according to score
top_movies = top_movies.reset_index()

# top_movies[['title', 'vote_count', 'vote_average', 'score']].head(20) # top 20 movies
t1 = top_movies[['title', 'score','id']].head(20)

print(t1['id'])


# In[7]:


# Distribution of average vote among movies in the dataset

fig = px.histogram(top_movies, x="vote_average")
fig.show()


# ### The following is also a simple recommender system but it's based on genres.
# 
# Steps involved:
# 1. Finding how many different types of genres are present.
# 2. Making a dictionary of genres (keys = genres name, values = list of movies belongs to the given genre)
# 3. Finaly sorting the values based on the score calculated above.

# In[7]:


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
            genres_based[x['name']] = pd.DataFrame(columns = top_movies.columns)
        genres_based[x['name']] = genres_based[x['name']].append(top_movies.iloc[i])  


# In[8]:


# Visualizing frequency of occurence of different genres

# Creating a count vector (list) containing frequency of a perticular genre
cnt = list()
for i in genres:
    cnt.append(genres_based[i].shape[0])
    
# Making a dataframe 
genre_cnt = pd.DataFrame( { 'genres' : list(genres),
                            'count'  : cnt
    
},
                         columns = ['genres','count']
)

fig = px.bar(genre_cnt, x='genres', y='count')
fig.show()


# In[9]:


print(top_movies)


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
for i in genres_based_rcmnd('War')['']:
    print(i)


# ## CONTENT BASED RECOMMENDER SYSTEM 
# 
# ### Metadata Based Recommender [ recommender based on movies keywords, cast, director(from crew dataset) and genres ]
# 
# This is more personalized. As this computes similarities between movies based on Movie Cast, Crew, Keywords and Genre.
# We will be using movie metadata (or content) to build this engine, this also known as Content Based Filtering. We will be using subset of movie due to limited computing power.
# 
# Steps involved:
# 1. Preprocessing the data:
#                   1.1 Cleaning the data ( manually verified and removed data with wrong format )
#                   1.2 Merging credits and keywords csv to the original dataset ie, movies_data
#                   1.3 Using only those movie which is present in links_small dataset ( it is a dataset which links movieid with   imdb and tmdbid )
#     
# 2. Creating a column name soup which contains director, cast, keywords and genres of the movie. Director name is written three times to give more weightage to the director and only top three cast name is taken.
# 
# 3. Then we used Count Vectorizer to create the count matrix and then we applied cosine similarity to calculate a numeric quantity that denotes the similarity between two movies.
# 
# 4. Finally, movies are recommended on the basis of cosine similarity values. Higher its value more is movie similar to that movie.

# In[10]:


# print(type(movies_data))


# In[11]:


#Preprocessing the data

movies_data['id'] = movies_data['id'].astype('int')  #The astype() function is used to cast a pandas object to a specified data type.

# merging both credits and keywords in movies_data on the basis of movie id
movies_data = movies_data.merge(credits, on='id')
movies_data = movies_data.merge(keywords, on='id')

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# taking only those movies whos id is present in link_small because of limited computing power
smd = movies_data[movies_data['id'].isin(links_small)]  
smd = smd.reset_index()

smd.head()


# In[12]:


def get_director(x):
    
    '''
    
    This function gives the name of first director occuring in the crew of the movie
    
    Parameters: x(list of dictionary): List containing name and corrosponding role of complete cast of the movie
    
    Returns: (string) It returns the first director name that appear in the list
    
    '''
    
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Applying literal_eval to get the right data type from the expression of string
smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['genres'] = smd['genres'].apply(literal_eval)

smd['director'] = smd['crew'].apply(get_director) 

# Taking all the movie cast in a list and then taking only the top 3 cast
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])# Strip Spaces and Convert to Lowercase

smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['genres'] = smd['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['genres'] = smd['genres'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x,x])  # giving more weight to the director relative to the entire cast


# In[13]:


def filter_keywords(x):
    
    '''
    
    This funtion remove those keywords which occur only once 
    
    Parameters: x(list): List containing keywords of the movie
    
    Returns: (list) It returns a list containg only those keywords which are present in keywords_count ( it is a dictionary containg those keywords which occur more than once )
    
    '''    
    
    words = []
    for i in x:
        if i in keywords_count.keys():
            words.append(i)
    return words


# Creating the count of every keywords
keywords_count = dict()
for i in range(len(smd['keywords'])):
    for j in range(len(smd['keywords'][i])):
        if smd['keywords'][i][j] not in keywords_count.keys():
            keywords_count[smd['keywords'][i][j]] = 0
        keywords_count[smd['keywords'][i][j]] +=1

# removing those keywords which occur only once
for i in list(keywords_count):
    if keywords_count[i] == 1:
        del keywords_count[i]


# In[14]:


# preprocessing

# Stemming the words 
stemmer = SnowballStemmer('english')

smd['keywords'] = smd['keywords'].apply(filter_keywords) # removing those keywords which occur only once
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# combining keywords, cast, director and genres
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
smd['soup'][0] 


# In[15]:


# Creating the Wordcloud for visualisation of the word which occur frequently in the dataset

# Combining all the text contained in smd['soup'] column
text = ""
for i in smd['soup']:
    text +=i
    
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

# Display the generated Word Cloud
# plot the WordCloud image                      
# plt.figure(figsize = (8, 8))
# plt.imshow(word_cloud)
# plt.axis("off")
 
plt.show()


# ### Definitions of terms used in the below code
# 
# 1. CountVectorizer is a tool provided by the scikit-learn library in Python. 
# 2. It is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.
# 3. stop_words='english' is a built-in list, ngram_range is just a string of n words in a row.
# 4. Cosine similarity is a measure of similarity, often used to measure document similarity in text analysis.

# In[16]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2) ,min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])  # Creating a mapping between movie and title and index


# In[17]:


df_cosine=pd.DataFrame(cosine_sim)
df_cosine


# In[20]:


# Creating Heatmap for visualization of correlation between different movies 
#specify size of heatmap
# fig, ax = plt.subplots(figsize=(8, 8))

# #create seaborn heatmap of only top 100 movies
# sns.heatmap(cosine_sim[:100,:100])


# In[21]:


# Graph structure to visualize similarity relation between selected movies

# g = nx.Graph()
# n = 10

# for i in range(n):
#     g.add_node(titles[i])

# for i in range(n):
#     for j in range(n):
#         if i != j and cosine_sim[i][j]>0:
#             g.add_edge(titles[i],titles[j],weight = cosine_sim[i][j])
    
# g = g.to_undirected()
# pos = nx.spring_layout(g)
# nx.draw_networkx_nodes(g, pos, node_size = 20)
# nx.draw_networkx_edges(g, pos,alpha = 0.3)
# nx.draw_networkx_labels(g, pos, font_size=10, horizontalalignment="right")

# plt.axis("off")
# plt.show()


# In[18]:


print((indices)) 
# print(indices['title'])
# for j in indices:
#     print(indices[j])


# In[19]:


def get_recommendations(title):    
    idx = indices[title] # movie id corrosponding to the given title 
    sim_scores = list(enumerate(cosine_sim[idx])) # list of cosine similarity scores value along the given index
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # sorting the given scores in ascending order
    sim_scores = sim_scores[1:201] # Taking only the top 100 scores
    movie_indices = [i[0] for i in sim_scores] # Finding the indices of 30 most similar movies
    return titles.iloc[movie_indices] 
# get_recommendations('Forrest Gump').head(10)
for i in get_recommendations('Criminal Law').head(20):
    print(i)


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

# In[20]:


# for i in ratings:
#     print(i)


# In[21]:


# drop the timestamp column since we dont need it now
ratings = ratings.drop('timestamp',axis=1)

#checking for missing values
ratings.isna().sum()

#check for the numbers of total movies and users
movies= ratings['movieId'].nunique()  #nunique is similar to count but only takes unique values
users=ratings['userId'].nunique()
print('total number of movies =', movies)
print('total number of users =', users)

#  HIstogram showing frequency of ratings given by different users
# fig = px.histogram(ratings, x="rating")
# fig.show()


# In[30]:


# columns to use for training
columns = ['userId','movieId','rating']

# create reader from surprise 
# the rating should lie in the provided scale
reader = Reader(rating_scale =(0.5,5))

#create dataset from dataframe
data = Dataset.load_from_df(ratings[columns],reader)

# create trainset ie the data which is present (ratings of those movies which are rated by respective users)
trainset = data.build_full_trainset()

# create testset, here the anti_testset is testset
# data containing users movie pairs which are not rated by that particular user
testset = trainset.build_anti_testset()
 

model = SVD(n_epochs = 25, verbose = False) #n_epochs:The number of iteration of the SGD(simple gradient descent) procedure. Default is 20
                                           #verbose:If True, prints the current epoch. Default is False.
    
cross_validate(model, data, measures=['RMSE','MAE'], cv= 5, verbose=False)
# print('Training Done')

#prediction
prediction = model.test(testset)


# ### Using user rating of selected movies to get suitable recommendations
# 
# 1. To incorporate the user in the system we inspected the dataset and found that mostly users have around 5-10 ratings.
# 2. To get recommendation according to taste of user we prompt the user to rate the selected movies.
# 3. We add those selection to our standerd dataset and run the collaborative filtering algorithm to get the ratings of other movies.
# 4. Finally, recommending top 10 movies based on highest ratings.

# In[35]:


example = { 'userId' : [99999,99999,99999,99999,99999],
           'movieId' : [31,1029,1293,1172,1343],
           'rating'  : [3.0, 4.5, 1.2, 3.3,2]
    
}

df = pd.DataFrame(example)
frames = [ratings, df]
result = pd.concat(frames)


# In[36]:


#create dataset from dataframe
data= Dataset.load_from_df(result[columns],reader)

#create trainset
trainset= data.build_full_trainset()

#create testset, here the anti_testset is testset
testset = trainset.build_anti_testset()

cross_validate(model,data, measures=['RMSE','MAE'], cv= 5, verbose= False) #cv is the number of parts in which data will be divided.
# print('Training Done')

#prediction
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
        for (iid,rating) in user_ratings:
            for i in range(movies_data.shape[0]):
                if movies_data['id'][i] == iid:
                    rcmnd.append([movies_data['id'][i],movies_data['title'][i]])
        break


# In[40]:


for i in rcmnd[:6]:
    print(i[0])


# ## Tkinter
# 
# Now we will create the interface for our recommender system using python Tkinter library. In this part we have used Genre based and Collaborative filltering based recommender system created in the above sections.
# 
# ### Basic guide to Tkinter
# 
# Tkinter is a python library which allow us to create GUIs.
# 
# The components of these GUIs can be arranged using different methods like grid method and frame method. In our code we have used grid method to arrange labels and fields in respective rows and columns. The position of label at a location is controlled by 'sticky' argument.
# 
# Apart from label, to get input from the user we have used spinbox and option menu ( i.e. dropdown ) from Tkinter.

# In[31]:


import tkinter as tk
import tkinter.ttk
from tkinter import *
import tkinter.messagebox


# In[32]:



l = [None for i in range(10)]

def genre_based():
    '''
       Callback function used for the submit button on the interface

       This function takes input directly from the Tkinter interface and based on the type of input provided it finds recommendation for the user and provide output on the Tkinter interface.
       Parameters: None
       Returns: None 

    ''' 

    for i in range(10):
        if l[i] is not None:
            l[i].grid_remove()
    event = clicked.get()

    # produced recommendations for the user based on the rating provided to selected movies if no genre is selected else recommend top movies from the selected genres
    if event == "Select Genre":
        rtts = list()
        rtts.append(l11.get())
        rtts.append(l12.get())
        rtts.append(l13.get())
        rtts.append(l14.get())
        rtts.append(l15.get())
        movie_ratings = [int(i) for i in rtts]
        print(movie_ratings)
        
        example = { 'userId' : [99999,99999,99999,99999,99999],
           'movieId' : [278, 13,637,122, 11],
           'rating'  : movie_ratings,
                  }
        
        df = pd.DataFrame(example)
        frames = [ratings, df]
        result = pd.concat(frames)
        
        #create dataset from dataframe
        data= Dataset.load_from_df(result[columns],reader)

        #create trainset
        trainset= data.build_full_trainset()

        #create testset, here the anti_testset is testset
        testset = trainset.build_anti_testset()

        cross_validate(model,data, measures=['RMSE','MAE'], cv= 5, verbose= True)
        print('Training Done')

        #prediction
        prediction = model.test(testset)
        prediction[99999]

        # An RMSE value of less than 2 is considered good
        #Now Recommend Users top 10 movies based on prediction

        from collections import defaultdict
        def get_top_n(prediction, n):

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
                for (iid,rating) in user_ratings:
                    for i in range(movies_data.shape[0]):
                        if movies_data['id'][i] == iid:
                            rcmnd.append([movies_data['id'][i],movies_data['title'][i]])
                break
    
        for i in range(min(10,len(rcmnd))):
            l[i] = Label(root ,  text = rcmnd[i][1])
            l[i].grid(row = 2+i, column = 5, sticky = W, pady = 5)                
    else:
        ll = list((genres_based[event][['title']].head(10))['title'])
        for i in range(min(10,len(ll))):
            l[i] = Label(root ,  text = ll[i])
            l[i].grid(row = 2+i, column = 5, sticky = W, pady = 5)


# In[33]:


# temp dataframe contains movies which are present in both ratings and movies_data dataframe

id_list = list(ratings['movieId'])
temp = movies_data.loc[movies_data['id'].isin(id_list)]
temp = temp.reset_index()
print(temp[['id' , 'title']])
temp['title'][1]


# In[34]:


# This snippet is used to find top ranked movies according to imdb score which are present in temp dataframe.

available_movies = []

for movie in list(t1['title']):
    movie = movie.lower()
    for i in range(temp.shape[0]):
        name = temp['title'][i].lower()
        if name == movie:
            available_movies.append((temp['id'][i] , movie))
            
print(available_movies)


# In[35]:


# create root window
root = Tk()

# root window title and dimension4
root.title("AZ Movie Recommender System")

# Set geometry (widthxheight)
root.geometry('1000x400')


# In[36]:


# Dropdown menu options
options = [
    'Action',
 'Adventure',
 'Animation',
 'Comedy',
 'Crime',
 'Documentary',
 'Drama',
 'Family',
 'Fantasy',
 'History',
 'Horror',
 'Music',
 'Mystery',
 'Romance',
 'Science Fiction',
 'TV Movie',
 'Thriller',
 'War',
 'Western'
]

# adding a label to the root window
l1 = Label(root, text = "MOVIE RECOMMENDER SYSTEM", fg = "blue")
l1.grid(row = 0,column = 1)
l2 = Label(root, text = "Select genre of the movie you want to watch : ")
l2.grid(row = 1,column = 0,sticky = W, pady = 2)

# datatype of menu text
clicked = StringVar()

# initial menu text
clicked.set( "Select Genre" )

# Create Dropdown menu
drop = OptionMenu( root , clicked , *options)
drop.grid(row = 3,column = 0, sticky = W)

# Creating seperators for better UI 
x1 = tkinter.ttk.Separator(root, orient=VERTICAL).grid(column=1, row=1, rowspan=12, sticky='ns')

l3 = Label(root, text = "Rate the following movies")
l3.grid(row = 1,column = 2,sticky = W, pady = 2)

# labels for movies name
l4 = Label(root, text="Movies Name").grid(row=2, column=2)
l5 = Label(root, text="the shawshank redemption").grid(row=3, column=2) # 278
l6 = Label(root, text="forest gump").grid(row=4, column=2) # 13
l7 = Label(root, text="life is beautiful").grid(row=5, column=2) #637
l8 = Label(root, text="the lord of the rings: the return of the king").grid(row=6, column=2) #122
l9 = Label(root, text="star wars").grid(row=7, column=2) # 11

# label for movies rating
l10 = Label(root, text="Rate the movie on the scale of 5").grid(row=2, column=3)
l11 = Spinbox(root, from_= 0, to = 5)
l11.grid(row=3, column=3)
l12 = Spinbox(root, from_= 0, to = 5)
l12.grid(row=4, column=3)
l13= Spinbox(root, from_= 0, to = 5)
l13.grid(row=5, column=3)
l14 = Spinbox(root, from_= 0, to = 5)
l14.grid(row=6, column=3)
l15 = Spinbox(root, from_= 0, to = 5)
l15.grid(row=7, column=3)


# button widget with green color text
button = Button(root, text = "SUBMIT" , fg = "white",bg = "green", command = genre_based)
button.grid(row = 9,column = 1, sticky = S)

# Creating seperators for better UI 
x2 = tkinter.ttk.Separator(root, orient=VERTICAL).grid(column=4, row=1, rowspan=12, sticky='ns')

l16 = Label(root, text="Results").grid(row=1, column= 5, sticky = W, pady=10)


# In[37]:


# start the program
root.mainloop()


# #  Additional Guidlines
#     
#     1. In this we have done content based recommender system using metadata such as cast, crew, keywords and genre.
#        You can also try making a movie recommender system using movies overviews and taglines in movies_data dataset.
#     
#     2. You can try using User-Based Collaborative filtering using Pearson Correlation.
#     
#        For better understanding of collaborative filtering use the beolow links: 
#        (REF - 1. https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)
#        
#     3. Instead of using SVD you can use other optimization algorithms like SVD++, K-NN, Slope one, etc. 
#        Also try tuning the algorithms parameters using GridSearchCV from scikit-learn.
#     
#        (REF - https://surpriselib.com/, https://realpython.com/build-recommendation-engine-collaborative-filtering/)
#        
#     4. For the collaborative filtering based recommender system we are tackling the cold-start problem for user by asking
#        ratings for fixed set of movies but this has obvious drawbacks. You can try tackling cold-start problem in better ways.

# In[ ]:




