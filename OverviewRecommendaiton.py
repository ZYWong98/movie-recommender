import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the CSV file
df = pd.read_csv("movieData.csv")  # Replace with the path to your file
