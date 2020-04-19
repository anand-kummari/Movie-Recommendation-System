class recommendation_methods():
    def __init__(self):
        pass

    def gaussian_filter(self,x, y, sigma):
        return math.exp(-(x-y)**2/(2*sigma**2))


    def entry_variables(self,df, id_entry):
        col_labels = []
        if pd.notnull(df['director_name'].iloc[id_entry]):
            for s in df['director_name'].iloc[id_entry].split('|'):
                col_labels.append(s)

        for i in range(3):
            column = 'actor_NUM_name'.replace('NUM', str(i+1))
            if pd.notnull(df[column].iloc[id_entry]):
                for s in df[column].iloc[id_entry].split('|'):
                    col_labels.append(s)

        if pd.notnull(df['plot_keywords'].iloc[id_entry]):
            for s in df['plot_keywords'].iloc[id_entry].split('|'):
                col_labels.append(s)
        return col_labels


    def add_variables(self,df, REF_VAR):
        for s in REF_VAR:
            df[s] = pd.Series([0 for _ in range(len(df))])
        colonnes = ['genres', 'actor_1_name', 'actor_2_name',
                    'actor_3_name', 'director_name', 'plot_keywords']
        for categorie in colonnes:
            for index, row in df.iterrows():
                if pd.isnull(row[categorie]):
                    continue
                for s in row[categorie].split('|'):
                    if s in REF_VAR:
                        df.set_value(index, s, 1)
        return df


    def recommand(self,df, id_entry):
        df_copy = df.copy(deep=True)
        liste_genres = set()
        # print(df['genres'].str.split('|').values)
        # for s in df['genres'].str.split('|').values:
        # print(df['genres'])
        for s in df['genres']:
            # print(s)
            for genre in s:
                # print(genre['name'])
                liste_genres = liste_genres.union(set(genre['name']))
        
        # Create additional variables to check the similarity
        variables = self.entry_variables(df_copy, id_entry)
        variables += list(liste_genres)
        df_new = self.add_variables(df_copy, variables)
        
        # determination of the closest neighbors: the distance is calculated / new variables
        X = df_new.as_matrix(variables)
        nbrs = NearestNeighbors(
            n_neighbors=31, algorithm='auto', metric='euclidean').fit(X)

        distances, indices = nbrs.kneighbors(X)
        xtest = df_new.iloc[id_entry].as_matrix(variables)
        xtest = xtest.reshape(1, -1)

        distances, indices = nbrs.kneighbors(xtest)

        return indices[0][:]

    def extract_parameters(self,df, liste_films):
        parametres_films = ['_' for _ in range(31)]
        i = 0
        max_users = -1
        for index in liste_films:
            parametres_films[i] = list(df.iloc[index][['movie_title', 'title_year',
                                                    'imdb_score', 'num_user_for_reviews',
                                                    'num_voted_users']])
            parametres_films[i].append(index)
            max_users = max(max_users, parametres_films[i][4])
            i += 1

        title_main = parametres_films[0][0]
        annee_ref = parametres_films[0][1]
        parametres_films.sort(key=lambda x: self.critere_selection(title_main, max_users,
                                                            annee_ref, x[0], x[1], x[2], x[4]), reverse=True)

        return parametres_films


    def sequel(self,titre_1, titre_2):
        if fuzz.ratio(titre_1, titre_2) > 50 or fuzz.token_set_ratio(titre_1, titre_2) > 50:
            return True
        else:
            return False


    def critere_selection(self,title_main, max_users, annee_ref, titre, annee, imdb_score, votes):
        if pd.notnull(annee_ref):
            facteur_1 = self.gaussian_filter(annee_ref, annee, 20)
        else:
            facteur_1 = 1

        sigma = max_users * 1.0

        if pd.notnull(votes):
            facteur_2 = self.gaussian_filter(votes, max_users, sigma)
        else:
            facteur_2 = 0

        if self.sequel(title_main, titre):
            note = 0
        else:
            note = imdb_score**2 * facteur_1 * facteur_2

        return note


    def add_to_selection(self,film_selection, parametres_films):
        film_list = film_selection[:]
        icount = len(film_list)
        for i in range(31):
            already_in_list = False
            for s in film_selection:
                if s[0] == parametres_films[i][0]:
                    already_in_list = True
                if self.sequel(parametres_films[i][0], s[0]):
                    already_in_list = True
            if already_in_list:
                continue
            icount += 1
            if icount <= 5:
                film_list.append(parametres_films[i])
        return film_list


    def remove_sequels(self,film_selection):
        removed_from_selection = []
        for i, film_1 in enumerate(film_selection):
            for j, film_2 in enumerate(film_selection):
                if j <= i:
                    continue
                if self.sequel(film_1[0], film_2[0]):
                    last_film = film_2[0] if film_1[1] < film_2[1] else film_1[0]
                    removed_from_selection.append(last_film)

        film_list = [film for film in film_selection if film[0]
                    not in removed_from_selection]

        return film_list


    def find_similarities(self,df, id_entry, del_sequels=True, verbose=False):
        if verbose:
            print(90*'_' + '\n' + "QUERY: films similar to id={} -> '{}'".format(id_entry,
                                                                                df.iloc[id_entry]['movie_title']))
        liste_films = self.recommand(df, id_entry)
        
        # Create a list of 31 films
        parametres_films = self.extract_parameters(df, liste_films)
        
        # Select 5 films from this list
        film_selection = []
        film_selection = self.add_to_selection(film_selection, parametres_films)
        
        # delation of the sequels
        if del_sequels:
            film_selection = self.remove_sequels(film_selection)
        
        # add new films to complete the list
        film_selection = self.add_to_selection(film_selection, parametres_films)
        #_____________________________________________
        selection_titres = []
        for i, s in enumerate(film_selection):
            selection_titres.append([s[0].replace(u'\xa0', u''), s[5]])
            if verbose:
                print("nº{:<2}     -> {:<30}".format(i+1, s[0]))

        return selection_titres
