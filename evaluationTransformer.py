def compute_recall_precision_for_star_wars(embeddings, titles, movie_title, ground_truth, top_k=5):
    # Find the index of the movie "Star Wars"
    try:
        movie_index = titles.index(movie_title)
    except ValueError:
        raise ValueError(f"'{movie_title}' not found in the dataset.")
    
    # Get recommendations
    similarity = cosine_similarity(embeddings[movie_index].reshape(1, -1), embeddings)
    similar_indices = similarity[0].argsort()[-top_k-1:][::-1]
    recommendations = [titles[i] for i in similar_indices if i != movie_index][:top_k]
    
    # Convert recommendations and ground_truth to binary vectors
    all_movies = set(titles)
    y_true = [1 if movie in ground_truth else 0 for movie in all_movies]
    y_pred = [1 if movie in recommendations else 0 for movie in all_movies]
    
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return precision, recall, recommendations

def main_with_star_wars_metrics(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    
    # Load and preprocess data
    data = load_data(file_path)
    embeddings, titles = generate_embeddings(data, bert_model, tokenizer, device)
    
    # Ground truth for Star Wars (example, replace with curated list or actual user data)
    ground_truth = [
        "The Empire Strikes Back",
        "Return of the Jedi",
        "Rogue One: A Star Wars Story",
        "The Force Awakens",
        "A New Hope"
    ]
    
    # Compute metrics for "Star Wars"
    movie_title = "Star Wars"
    precision, recall, recommendations = compute_recall_precision_for_star_wars(embeddings, titles, movie_title, ground_truth)
    
    print(f"Recommendations for '{movie_title}': {recommendations}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

# Run the evaluation
file_path = "movies.csv"  # Replace with your actual file path
main_with_star_wars_metrics(file_path)
