import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Define the Transformer for movie recommendations
class MovieRecommenderTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(MovieRecommenderTransformer, self).__init__()
        
        # Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
    
    def forward(self, embeddings):
        # Transformer expects input [seq_len, batch_size, embed_dim]
        x = embeddings.permute(1, 0, 2)
        output = self.transformer(x)
        return output.permute(1, 0, 2)  # Convert back to [batch_size, seq_len, embed_dim]

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Generate embeddings using BERT
def generate_embeddings(data, model, tokenizer, device):
    titles = data['title'].tolist()
    overviews = data['overview'].tolist()
    
    embeddings = []
    for overview in overviews:
        inputs = tokenizer(overview, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())  # CLS token embedding
    return torch.tensor(embeddings).squeeze(1), titles

# Recommend movies based on similarity
def recommend_movies(embeddings, titles, movie_index, top_k=5):
    similarity = cosine_similarity(embeddings[movie_index].reshape(1, -1), embeddings)
    similar_indices = similarity[0].argsort()[-top_k-1:][::-1]  # Exclude the input movie itself
    return [titles[i] for i in similar_indices if i != movie_index][:top_k]

# Main function
def main(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    
    # Load and preprocess data
    data = load_data(file_path)
    embeddings, titles = generate_embeddings(data, bert_model, tokenizer, device)
    
    # Initialize Transformer model
    embed_dim = embeddings.shape[1]
    model = MovieRecommenderTransformer(embed_dim, num_heads=4, num_layers=2, ff_dim=256).to(device)
    
    # Example: Recommend movies for the first entry
    movie_index = 0  # Change as needed
    recommendations = recommend_movies(embeddings, titles, movie_index)
    
    print(f"Recommendations for '{titles[movie_index]}':")
    for rec in recommendations:
        print(f"- {rec}")

# Run the script with your file
file_path = "movies.csv"  # Replace with your actual file path
main(file_path)
