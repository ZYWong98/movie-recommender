
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

