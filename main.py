#%%
import pandas as pd
import json
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

# %%
