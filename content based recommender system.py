#conrec is the function name

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
# ## CONTENT BASED RECOMMENDER SYSTEM
#
# ### Metadata Bas
# ed
# Recommender[recommender
# based
# on
# movies
# keywords, cast, director(
# from crew dataset) and genres]
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

# In[12]:


print(type(movies_data))

# In[13]:


# Preprocessing the data

movies_data['id'] = movies_data['id'].astype(
    'int')  # The astype() function is used to cast a pandas object to a specified data type.

# merging both credits and keywords in movies_data on the basis of movie id
movies_data = movies_data.merge(credits, on='id')
movies_data = movies_data.merge(keywords, on='id')

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# taking only those movies whos id is present in link_small because of limited computing power
smd = movies_data[movies_data['id'].isin(links_small)]
smd = smd.reset_index()

smd.head()


# In[14]:


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
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
smd['cast'] = smd['cast'].apply(
    lambda x: [str.lower(i.replace(" ", "")) for i in x])  # Strip Spaces and Convert to Lowercase

smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['genres'] = smd['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['genres'] = smd['genres'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(
    lambda x: [x, x, x])  # giving more weight to the director relative to the entire cast


# In[15]:


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
        keywords_count[smd['keywords'][i][j]] += 1

# removing those keywords which occur only once
for i in list(keywords_count):
    if keywords_count[i] == 1:
        del keywords_count[i]

# In[16]:


# preprocessing

# Stemming the words
stemmer = SnowballStemmer('english')

smd['keywords'] = smd['keywords'].apply(filter_keywords)  # removing those keywords which occur only once
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# combining keywords, cast, director and genres
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
# smd['soup'][0]

# In[17]:


# Creating the Wordcloud for visualisation of the word which occur frequently in the dataset

# Combining all the text contained in smd['soup'] column
text = ""
for i in smd['soup']:
    text += i

word_cloud = WordCloud(collocations=False, background_color='white').generate(text)

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

# In[18]:


count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])  # Creating a mapping between movie and title and index

# In[19]:


df_cosine = pd.DataFrame(cosine_sim)
# df_cosine

# In[20]:


# Creating Heatmap for visualization of correlation between different movies
# specify size of heatmap
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


# In[22]:


# print((indices))


# print(indices['title'])
# for j in indices:
#     print(indices[j])


# In[23]:


def get_recommendations(title):
    idx = indices[title]  # movie id corrosponding to the given title
    sim_scores = list(enumerate(cosine_sim[idx]))  # list of cosine similarity scores value along the given index
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # sorting the given scores in ascending order
    sim_scores = sim_scores[1:201]  # Taking only the top 100 scores
    movie_indices = [i[0] for i in sim_scores]  # Finding the indices of 30 most similar movies
    return titles.iloc[movie_indices]


# get_recommendations('Forrest Gump').head(10)
def conrec(name):
    for i in get_recommendations(name).head(20):
        print(i['id'])

