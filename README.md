# movie-recommender


(overview or genre+keywords) to do our Content-based filtering system
add final row to dataset describing what kind of movies user wants (make sure all fields have input, are preprocessed)
 
for overview recommendation
use TFIDFVectorizer from scikit-learn to fit, transform overview column (make sure everything is preprocessed)
use liner_kernel() to compute cosine similarity (find similarities between 2 movies)
create reverse of indices and map movie titles (map title as index)
make recommendation function
				-- get movie index
				-- get similarity score of movies linked to that movie
				-- sort movies by similarity score
				-- filter top 10 movies based on score
				-- get movie indices and return top 10 movies 

for genre+keyword recommendation
clean the data again of stopwords, make words lowercase(just in case)
create 'metadata soup' (combine both genre and keywords into string of words to feed vectorizer)
use CountVectorizer from scikit-learn to fit soup and obtain matrix
use cosine similarity function to calculate score (fit count matrix twice)
create reverse of indices and map movie titles (map title as index)
make recommendation function (same function as above but replace consine similarity matrix)
