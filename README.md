# movie-recommender

![Screenshot 2024-11-02 at 10 43 20 AM](https://github.com/user-attachments/assets/f6615393-9556-47bc-ac50-231b4b9be581)

![Screenshot 2024-11-02 at 10 42 56 AM](https://github.com/user-attachments/assets/99b5123e-07f9-454b-9c9f-b0063e30be1b)

## Step 4 :  Model Design and implementation and training - Due November 4th
  
	* (overview or genre+keywords) to do our Content-based filtering system
		* add final row to dataset describing what kind of movies user wants (make sure all fields have input, are preprocessed)
	 
	* for overview recommendation
		* use TFIDFVectorizer from scikit-learn to fit, transform overview column (make sure everything is preprocessed)
		* use liner_kernel() to compute cosine similarity (find similarities between 2 movies)
		* create reverse of indices and map movie titles (map title as index)
		* make recommendation function
			* get movie index
			* get similarity score of movies linked to that movie
			* sort movies by similarity score
			* filter top 10 movies based on score
			* get movie indices and return top 10 movies 
	
	* for genre+keyword recommendation
		* clean the data again of stopwords, make words lowercase(just in case)
		* create 'metadata soup' (combine both genre and keywords into string of words to feed vectorizer)
		* use CountVectorizer from scikit-learn to fit soup and obtain matrix
		* use cosine similarity function to calculate score (fit count matrix twice)
		* create reverse of indices and map movie titles (map title as index)
		* make recommendation function (same function as above but replace consine similarity matrix)

<br />

## Step 5 : Evaluation and Analysis : 11/11/2024

* Objectives:
	* Relevance and Accuracy of Recommendations
		* Precision: Evaluate the proportion of recommended movies that are relevant to the user. You can measure this by checking if the top recommendations are indeed similar to the user’s preferred genre and keywords.
		* User Testing: Gather feedback from users on the recommendations, especially in terms of relevance to the user’s interests. Ask users to rate the similarity of recommended movies to their original preferences.
		* Recall: Check how well the system retrieves all relevant movies, particularly for niche genres or unique keyword combinations.
	* A/B Testing of Similarity Models
		* Alternative Models: Try using different similarity metrics (like Jaccard similarity for categorical data) or model configurations, then conduct A/B testing to compare results with the cosine similarity approach. This comparison can reveal which model yields the most relevant recommendations.
	 * Impact of Metadata Soup Design
		* Feature Importance: Evaluate how well the “metadata soup” (combined genre and keywords) captures meaningful information. You could compare results using different metadata, such as testing genre-only or keyword-only recommendations to see how each contributes to the final recommendations.
	 * Quality of Similarity Scores
		* Threshold Testing: Experiment with different cosine similarity thresholds to see if certain cut-offs yield more accurate or relevant recommendations.

<br />

## Step 6 : Finalization and Deployment : 11/18/2024

* Objectives:
	* x	

<br />

## Step 7 : Presentation and Report : 11/25/2024

* Objectives:
	* x
 
<br />




