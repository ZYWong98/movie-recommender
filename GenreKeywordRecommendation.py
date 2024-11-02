""" 
Objective for genre+keyword recommendation: 
    clean the data again of stopwords, make words lowercase(just in case)
    create 'metadata soup' (combine both genre and keywords into string of words to feed vectorizer)
    use CountVectorizer from scikit-learn to fit soup and obtain matrix
    use cosine similarity function to calculate score (fit count matrix twice)
    create reverse of indices and map movie titles (map title as index)
    make recommendation function (same function as above but replace consine similarity matrix)
"""

#Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Clean Data
# Assuming df is your DataFrame and contains genre and keywords columns.
# Remove stopwords, make words lowercase, and clean up genre and keywords.
# Ensure both columns contain strings of words.
def clean_data(x):
    if isinstance(x, str):
        return x.lower()
    else:
        return ''

df['genre'] = df['genre'].apply(clean_data)
df['keywords'] = df['keywords'].apply(clean_data)

# Create ‘Metadata Soup’
# Combine genre and keywords columns into a single “metadata soup.”

df['metadata_soup'] = df['genre'] + ' ' + df['keywords']

# Use CountVectorizer to Fit Metadata Soup
# Initialize and fit CountVectorizer on the combined metadata.

count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(df['metadata_soup'])

# Calculate Cosine Similarity
# Fit the count matrix twice to get cosine similarity.
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# Create Reverse Mapping of Indices
# Create a dictionary to map movie titles to their indices.
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Build Recommendation Function
# Define a recommendation function that uses the cosine similarity matrix.
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]  # Get the index of the movie that matches the title
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]


# Usage
# To get recommendations for a specific movie:
recommendations = get_recommendations("Movie Title")
print(recommendations)
























