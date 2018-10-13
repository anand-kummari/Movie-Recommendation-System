# Movie-Recommendation-System
This project aims at building a **recommendation engine** from around **5000 movies** and **TV series**.When the user has provided the name of a film he liked, the engine should be able to select in the database a list of 5 films that the user will enjoy.


In general recommendation engines are of three types:
1. **Popularity-based**: easy to implement
2. **content-based**: uses user descriptions for recommendations
3. **Collaborative filtering** : record various user activities and recommend based on user similarities


This project aims to build recommendation engines that are `popularity-based` and `content-based`

### About the DataSet
* I am using TMDB 5000 dataset.
* It contains two csv files namely `tmdb_5000_credits.csv` and `tmdb_5000_movies.csv.
* credits file contains cast and crew details i.e., cast_id,character,credit_id,gender,id,name,department,job_name.
* movies file contains budget,genres,homepage,keywords,language,release year,popularity,production company,revenue,title,vote average,vote count etc.`

## Pre-processing the DataSet
1. Merge the two files to form a data structure that contains required details on movie like director ,lead actors,title,keywords,release date,language,genre,budget,vote count etc.
2. List the keywords present in the dataset.
3. Count the number of times each keyword appears.
4. set any missing information (if required)


## Recommendation engine
1. Determine N  films with a content similar to the entry provided by the user
    * From the user description get kekywords,director,actors,genre and other important fields.
    * Build a matrix which has the above mentioned fields as columns and films as rows.
    * In this matrix, the  Aij  coefficients take either 0 or 1 depending on the correspondance between the significance of           column j and the content of film i
    * Use **root mean square(RMS)** to determine the distance between the `i`th film and `j`th film.

2. select the 5 most popular films among these  N  films and recommend to the user.
