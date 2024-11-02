# movie-recommender


(overview or genre+keywords) to do our Content-based filtering system
1. add final row to dataset describing what kind of movies user wants (make sure all fields have input, are preprocessed)
 
for overview recommendation
2. use TFIDFVectorizer from scikit-learn to fit, transform overview column (make sure everything is preprocessed)
3. use liner_kernel() to compute cosine similarity (find similarities between 2 movies)
4. create reverse of indices and map movie titles (map title as index)
5. make recommendation function
	-- get movie index
	-- get similarity score of movies linked to that movie
	-- sort movies by similarity score
	-- filter top 10 movies based on score
	-- get movie indices and return top 10 movies 

for genre+keyword recommendation
2. clean the data again of stopwords, make words lowercase(just in case)
3. create 'metadata soup' (combine both genre and keywords into string of words to feed vectorizer)
4. use CountVectorizer from scikit-learn to fit soup and obtain matrix
5. use cosine similarity function to calculate score (fit count matrix twice)
6. create reverse of indices and map movie titles (map title as index)
7. make recommendation function (same function as above but replace consine similarity matrix)



![Screenshot 2024-11-02 at 10 43 20 AM](https://github.com/user-attachments/assets/f6615393-9556-47bc-ac50-231b4b9be581)

![Screenshot 2024-11-02 at 10 42 56 AM](https://github.com/user-attachments/assets/99b5123e-07f9-454b-9c9f-b0063e30be1b)


