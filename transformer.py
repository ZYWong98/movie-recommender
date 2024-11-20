import torch
import torch.nn as nn
import torch.optim as optim

class MovieRecommenderTransformer(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(MovieRecommenderTransformer, self).__init__()
        
        # Embeddings for users and movies
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        
        # Transformer layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(embed_dim, num_movies)
        
    def forward(self, user_ids, movie_ids):
        # Embed users and movies
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        
        # Concatenate and reshape for the transformer
        x = torch.cat((user_embed.unsqueeze(1), movie_embed), dim=1)  # [batch_size, seq_len, embed_dim]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, embed_dim]
        
        # Pass through the transformer
        x = self.transformer(x)
        
        # Take the output corresponding to the user representation
        user_output = x[0]  # [batch_size, embed_dim]
        
        # Predict scores for all movies
        scores = self.output_layer(user_output)
        
        return scores

# Example usage
num_users = 1000
num_movies = 5000
embed_dim = 128
num_heads = 4
num_layers = 2
ff_dim = 256

model = MovieRecommenderTransformer(num_users, num_movies, embed_dim, num_heads, num_layers, ff_dim)

# Dummy data
batch_size = 32
user_ids = torch.randint(0, num_users, (batch_size,))
movie_ids = torch.randint(0, num_movies, (batch_size, 10))  # 10 movies per user

# Forward pass
scores = model(user_ids, movie_ids)

# Recommended movie indices for each user
recommendations = torch.argmax(scores, dim=1)

print("Recommended movies:", recommendations)
