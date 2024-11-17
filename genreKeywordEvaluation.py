import pandas as pd
import re
from evaluation_metrics import precision_at_k, recall_at_k
from GenreKeywordRecommendation import get_recommendations
# Function to standardize movie titles by removing spaces, special characters, and converting to lowercase
def standardize_title(title):
    title = title.lower()                    # Convert to lowercase
    title = re.sub(r'[^a-zA-Z0-9]', '', title)  # Remove special characters and spaces
    return title

# Step 1: Get recommendations for a specific movie
movie_title = "the godfather"
recommended_movies = get_recommendations(movie_title)

# Apply standardization to recommended_movies
recommended_movies = recommended_movies.apply(standardize_title)

# Step 2: Define relevant movies and apply standardization
relevant_movies = pd.Series([" the godfather: part iii", "the empire strikes back", "return of the jedi"], name='movie')
relevant_movies = relevant_movies.apply(standardize_title)

# Step 3: Create a DataFrame to represent recommendation results
# Combine both lists, remove duplicates, and reset index
all_movies = pd.Series(pd.concat([recommended_movies, relevant_movies]).unique(), name='movie')
df = pd.DataFrame({'movie': all_movies})

# Add columns indicating whether each movie is recommended and relevant
df['y_recommended'] = df['movie'].isin(recommended_movies)
df['y_actual'] = df['movie'].isin(relevant_movies)

# Step 4: Set the value for k
k = 5  # Adjust as desired

# Step 5: Calculate Precision@K and Recall@K
precision = precision_at_k(df, k, y_test='y_actual', y_pred='y_recommended')
recall = recall_at_k(df, k, y_test='y_actual', y_pred='y_recommended')

# Step 6: Output the results
print(f"Precision@{k}: {precision}")
print(f"Recall@{k}: {recall}")
