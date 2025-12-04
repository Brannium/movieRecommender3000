from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
import os
from datahandler import DataHandler
from recommender import MovieRecommender

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
MOVIES_FILE = "data/movies.csv"
RATINGS_FILE = "data/interactions.csv"
TITLES_FILE = "../dataset/TMDB_movie_dataset_v11.csv"

# Load data at app startup
data_handler = DataHandler(MOVIES_FILE, RATINGS_FILE, TITLES_FILE)

# Shared ratings DataFrame - stores new ratings during session
# Columns: user_id, movie_id, rating
ratings_df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])

# Global recommender instance (lazy-initialized)
recommender = None

def get_all_movies_with_titles():
    """
    Merge movie IDs with their titles for display
    Returns DataFrame with columns: id, title
    """
    movies_df = data_handler.getAllMovies()[['id']].copy()
    titles_df = data_handler.getMovieTitles()[['id', 'title']].copy()
    merged = movies_df.merge(titles_df, on='id', how='left')
    merged['title'] = merged['title'].fillna('Unknown')
    return merged


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/movies', methods=['GET'])
def get_movies():
    """
    Get paginated and filtered movies
    Query params:
      - page (int): page number (1-indexed), default=1
      - limit (int): items per page, default=50
      - search (str): partial title search, optional
    """
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    search_query = request.args.get('search', '', type=str).lower()
    
    # Get all movies with titles
    movies = get_all_movies_with_titles()
    
    # Filter by search query if provided
    if search_query:
        movies = movies[movies['title'].str.lower().str.contains(search_query, na=False)]
    
    # Calculate pagination
    total = len(movies)
    total_pages = (total + limit - 1) // limit  # Ceiling division
    
    # Validate page number
    if page < 1:
        page = 1
    if page > total_pages and total_pages > 0:
        page = total_pages
    
    # Get page slice
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    page_movies = movies.iloc[start_idx:end_idx]
    
    # Convert to list of dicts for JSON response
    movies_list = [
        {'id': int(row['id']), 'title': row['title']}
        for _, row in page_movies.iterrows()
    ]
    
    return jsonify({
        'movies': movies_list,
        'page': page,
        'limit': limit,
        'total': total,
        'total_pages': total_pages
    })


@app.route('/api/rate', methods=['POST'])
def submit_rating():
    """
    Submit a movie rating
    Expects JSON: {user_id: int, movie_id: int, rating: int}
    rating should be: -1 (dislike), 1 (like), 2 (superlike)
    """
    global ratings_df
    
    data = request.get_json()
    
    # Validate input
    if not data or 'user_id' not in data or 'movie_id' not in data or 'rating' not in data:
        return jsonify({'error': 'Missing required fields: user_id, movie_id, rating'}), 400
    
    user_id = data['user_id']
    movie_id = data['movie_id']
    rating = data['rating']
    
    # Validate rating value
    if rating not in [-1, 1, 2]:
        return jsonify({'error': 'Rating must be -1 (dislike), 1 (like), or 2 (superlike)'}), 400
    
    # Add to ratings DataFrame
    new_rating = pd.DataFrame({
        'user_id': [user_id],
        'movie_id': [movie_id],
        'rating': [rating]
    })
    ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
    
    return jsonify({
        'success': True,
        'message': f'Rating submitted for movie {movie_id}',
        'rating_count': len(ratings_df)
    })


@app.route('/api/ratings', methods=['GET'])
def get_ratings():
    """
    Get all ratings submitted during this session
    Returns the shared ratings DataFrame as JSON
    """
    return jsonify({
        'ratings': ratings_df.to_dict(orient='records'),
        'count': len(ratings_df)
    })


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """
    Get movie recommendations based on rated movies
    Expects JSON: {user_id: int, top_k: int (optional, default=20)}
    Uses rated movies from the ratings_df to train/run recommender
    """
    global recommender
    
    try:
        data = request.get_json()
        user_id = data.get('user_id', 1)
        top_k = data.get('top_k', 20)
        
        # Validate that there are rated movies
        if len(ratings_df) < 20:
            return jsonify({
                'error': 'Not enough rated movies. Please rate at least 20 movies.',
                'recommendations': []
            }), 400
        
        # Initialize recommender if not already done
        if recommender is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            recommender = MovieRecommender(
                allMovies=data_handler.getAllMovies(),
                userRatings=ratings_df,
                movieTitles=data_handler.getMovieTitles(),
                device=device
            )
            print("Training recommender model...")
            recommender.train_recommender()
            print("Recommender training complete.")
        else:
            # Update recommender with new ratings
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            recommender = MovieRecommender(
                allMovies=data_handler.getAllMovies(),
                userRatings=ratings_df,
                movieTitles=data_handler.getMovieTitles(),
                device=device
            )
            print("Retraining recommender model with new ratings...")
            recommender.train_recommender()
            print("Recommender retraining complete.")
        
        # Get recommendations
        recommendations = recommender.recommend_for_user(user_id, top_k=top_k)
        
        # Format recommendations with titles
        rec_list = []
        for movie_id, score in recommendations:
            title = data_handler.getMovieTitle(movie_id)
            rec_list.append({
                'id': int(movie_id),
                'title': title,
                'score': float(score)
            })
        
        return jsonify({
            'recommendations': rec_list,
            'count': len(rec_list),
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Error generating recommendations: {str(e)}',
            'recommendations': []
        }), 500


def get_ratings_df():
    """
    Export function for external access to the ratings DataFrame
    Can be imported and used by other modules (e.g., recommender.py)
    """
    return ratings_df


if __name__ == '__main__':
    print("Starting Flask Movie Rater...")
    print(f"Movies loaded: {len(data_handler.getAllMovies())}")
    print(f"Movie titles loaded: {len(data_handler.getMovieTitles())}")
    app.run(debug=True, port=5000)
