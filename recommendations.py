#%%
import json
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math, nltk, warnings
from nltk.corpus import wordnet
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from wordcloud import WordCloud, STOPWORDS
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "last_expr"
# pd.options.display.max_columns = 50
%mpl inline
warnings.filterwarnings('ignore')
PS = nltk.stem.PorterStemmer()

#%%
def loadMovies(file):
    df = pd.read_csv(file)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries',
                    'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

#%%
def loadCredits(file):
    df = pd.read_csv(file)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

#%%
def get_name(list, values):
    result = list
    try:
        for idx in values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan


#%%
def get_director(crew_data):
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return get_name(directors, [0])

def select_name(keywords):
    return '|'.join([x['name'] for x in keywords])

def Movie_data(movies, credits):
    tmdb_movies = movies.copy()
    tmdb_movies.rename(columns=new_columns, inplace=True)
    tmdb_movies['year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: get_name(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: get_name(x, [0, 'name']))
    tmdb_movies['director'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1'] = credits['cast'].apply(lambda x: get_name(x, [1, 'name']))
    tmdb_movies['actor_2'] = credits['cast'].apply(lambda x: get_name(x, [2, 'name']))
    tmdb_movies['actor_3'] = credits['cast'].apply(lambda x: get_name(x, [3, 'name']))
    tmdb_movies['keywords'] = tmdb_movies['keywords'].apply(select_name)
    return tmdb_movies

#%%
new_columns = {
    'budget': 'budget',
    'genres': 'genres',
    'revenue': 'gross',
    'title': 'movie_title',
    'runtime': 'duration',
    'original_language': 'language',
    'keywords': 'keywords',
    'vote_count': 'num_voted_users'}

#%%
# load the dataset
credits = loadCredits("tmdb_5000_credits.csv")
movies = loadMovies("tmdb_5000_movies.csv")
df_initial = Movie_data(movies, credits)
print('Shape:',df_initial.shape)
#%%
# info on variable types, null values and their percentage
tab_info=pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values'}))
tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.rename(index={0:'null values (%)'}))
tab_info

#%%
# create a keyword set from the csv
keyword_set = set()
for item in df_initial['keywords'].str.split('|').values:
    if isinstance(item, float) and pd.isnull(item):
        print(item)
        continue  # only happen if item = NaN
    keyword_set = keyword_set.union(item)

# remove null chain entry
keyword_set.remove('')
#%%
print(type(keyword_set))
for s in keyword_set:
    print(s)
    break
#%%
def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for keyword in df[ref_col].str.split('|'):        
        if type(keyword) == float and pd.isnull(keyword):
            continue 
        for s in [s for s in keyword if s in liste]: 
            if pd.notnull(s): keyword_count[s] += 1
    
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

#%%
keyword_occurences, count = count_word(df_initial, 'keywords', keyword_set)
#%%
# code snippet to show keyword occurences in wordcloud representation
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)

fig = plt.figure(1, figsize=(18,13))
ax1 = fig.add_subplot(2,1,1)

# define the dictionary used to produce the wordcloud
words = dict()
trunc_occurences = keyword_occurences[0:50]
for s in trunc_occurences:
    words[s[0]] = s[1]

# define the color of the words
tone = 55.0

wordcloud = WordCloud(width=1000,height=300, background_color='black', 
                      max_words=1628,relative_scaling=1,
                      color_func = random_color_func,
                      normalize_plurals=False)
wordcloud.generate_from_frequencies(words)
ax1.imshow(wordcloud, interpolation="bilinear")
ax1.axis('off')
# fig.savefig('wordcloud.png')