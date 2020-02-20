#%%
import json
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math, nltk, warnings
# from nltk.corpus import wordnet
nltk.download('wordnet')
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
%matplotlib inline
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

wordcloud = WordCloud(width=800,height=500, background_color='black', 
                      max_words=1628,relative_scaling=1,
                      color_func = random_color_func,
                      normalize_plurals=False)
wordcloud.generate_from_frequencies(words)
ax1.imshow(wordcloud, interpolation="bilinear")
ax1.axis('off')
# fig.savefig('wordcloud.png')
#%%
# get filling factor and missing count in dataframe
missing_df = df_initial.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_count']
missing_df['filling_factor'] = (df_initial.shape[0]-missing_df['missing_count'])/df_initial.shape[0] * 100
missing_df.sort_values('filling_factor').reset_index(drop = True)

#%%
df_initial['decade'] = df_initial['year'].apply(lambda x:((x-1900)//10)*10)
# function that extract statistical parameters from a groupby objet:
def get_stats(gr):
    return {'min':gr.min(),'max':gr.max(),'count': gr.count(),'mean':gr.mean()}
# Creation of a dataframe with statitical infos on each decade:
test = df_initial['year'].groupby(df_initial['decade']).apply(get_stats).unstack()
#%%
sns.set_context("poster", font_scale=0.85)
# funtion used to set the labels
def label(s):
    val = (1900 + s, s)[s < 100]
    chain = '' if s < 50 else "{}'s".format(int(val))
    return chain

plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(11, 6))
labels = [label(s) for s in  test.index]
sizes  = test['count'].values
explode = [0.2 if sizes[i] < 100 else 0.01 for i in range(11)]
ax.pie(sizes, explode = explode, labels=labels,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow=False, startangle=0)
ax.axis('equal')
ax.set_title('% of films per decade',
             bbox={'facecolor':'k', 'pad':5},color='w', fontsize=16);
df_initial.drop('decade', axis=1, inplace = True)
# f.savefig('pieChart.png')

#%%
df_duplicate_cleaned = df_initial
#%%
# Grouping by roots
# Collect the keywords
def keywords_inventory(dataframe, coloumn = 'keywords'):
    PS = nltk.stem.PorterStemmer()
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys = []
    icount = 0
    for s in dataframe[coloumn]:
        if pd.isnull(s): continue
        for t in s.split('|'):
            t = t.lower() ; racine = PS.stem(t)
            if racine in keywords_roots:                
                keywords_roots[racine].add(t)
            else:
                keywords_roots[racine] = {t}
    
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)            
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    print("Nb of keywords in variable '{}': {}".format(coloumn,len(category_keys)))
    return category_keys, keywords_roots, keywords_select

keywords, keywords_roots, keywords_select = keywords_inventory(df_duplicate_cleaned,
                                                               coloumn = 'keywords')

#%%
# Replacement of the keywords by the main form
def replacement_df_keywords(df, dico_remplacement, roots = False):
    df_new = df.copy(deep = True)
    for index, row in df_new.iterrows():
        chaine = row['keywords']
        if pd.isnull(chaine): continue
        nouvelle_liste = []
        for s in chaine.split('|'): 
            clef = PS.stem(s) if roots else s
            if clef in dico_remplacement.keys():
                nouvelle_liste.append(dico_remplacement[clef])
            else:
                nouvelle_liste.append(s)       
        df_new.at[index, 'keywords'] = '|'.join(nouvelle_liste) 
        # print(df_new['keywords'].iloc[index])#,'keywords'))
        # print(nouvelle_liste)
        # break
    return df_new

# Replacement of the keywords by the main keyword
df_keywords_cleaned = replacement_df_keywords(df_duplicate_cleaned, keywords_select,
                                               roots = True)

#%%
# get the synomyms of the word 'word'
def get_synonymes(word):
    lemma = set()
    for ss in wordnet.synsets(word):
        for w in ss.lemma_names():
            # just get the 'nouns':
            index = ss.name().find('.')+1
            if ss.name()[index] == 'n': lemma.add(w.lower().replace('_',' '))
    return lemma  

def test_keyword(mot, key_count, threshold):
    return (False , True)[key_count.get(mot, 0) >= threshold]

#%%
keyword_occurences.sort(key = lambda x:x[1], reverse = False)
key_count = dict()
for s in keyword_occurences:
    key_count[s[0]] = s[1]

# Creation of a dictionary to replace keywords by higher frequency keywords
remplacement_mot = dict()
icount = 0
for index, [mot, nb_apparitions] in enumerate(keyword_occurences):
    if nb_apparitions > 5: continue  # only the keywords that appear less than 5 times
    lemma = get_synonymes(mot)
    if len(lemma) == 0: continue     # case of the plurals
    #_________________________________________________________________
    liste_mots = [(s, key_count[s]) for s in lemma 
                  if test_keyword(s, key_count, key_count[mot])]
    liste_mots.sort(key = lambda x:(x[1],x[0]), reverse = True)    
    if len(liste_mots) <= 1: continue       # no replacement
    if mot == liste_mots[0][0]: continue    # replacement by himself
    icount += 1
    if  icount < 8:
        print('{:<12} -> {:<12} (init: {})'.format(mot, liste_mots[0][0], liste_mots))    
    remplacement_mot[mot] = liste_mots[0][0]

print(90*'_'+'\n'+'The replacement concerns {}% of the keywords.'
      .format(round(len(remplacement_mot)/len(keywords)*100,2)))

#%%
# 2 successive replacements
print('Keywords that appear both in keys and values:'.upper()+'\n'+45*'-')
icount = 0
for s in remplacement_mot.values():
    if s in remplacement_mot.keys():
        icount += 1
        if icount < 10: print('{:<20} -> {:<20}'.format(s, remplacement_mot[s]))

for key, value in remplacement_mot.items():
    if value in remplacement_mot.keys():
        remplacement_mot[key] = remplacement_mot[value] 

#%%
# replacement of keyword varieties by the main keyword
df_keywords_synonyms = \
            remplacement_df_keywords(df_keywords_cleaned, remplacement_mot, roots = False)   
keywords, keywords_roots, keywords_select = \
            keywords_inventory(df_keywords_synonyms, coloumn = 'keywords')