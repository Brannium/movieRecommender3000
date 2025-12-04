# MovieSelector3000 - Flask Movie Rating UI

A Flask + HTML/JavaScript web application for rating movies. Features pagination, full-text search, and persistent rating storage.

## Features

- ✨ **Browse Movies** - Display movies in paginated lists (50 per page)
- 🔍 **Search** - Real-time search with partial title matching
- 👍 **Rate Movies** - Three rating options: Dislike (-1), Like (1), SuperLike (2)
- 🤖 **AI Recommendations** - Get personalized recommendations based on your ratings (uses trained ML model)
- 💾 **Persistent Storage** - All ratings saved to a shared DataFrame accessible by other modules
- 🎨 **Modern UI** - Responsive design with smooth interactions
- ⚡ **Fast** - Efficient API with client-side pagination and caching

## Setup

### 1. Install Dependencies

```bash
cd movie-recommender/movieselector3000
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure these files exist in the `dataset/` directory:
- `movies_stripped.csv` - Contains movie IDs
- `interactions.csv` - User ratings (columns: user_id, movie_id, rating)
- `TMDB_movie_dataset_v11.csv` - Movie titles and metadata

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

1. **Open the Web UI** - Navigate to `http://localhost:5000` in your browser
2. **Search Movies** - Type in the search box to find movies by title (partial matching supported)
3. **Rate Movies** - Click buttons next to each movie:
   - 👎 **Dislike** - Rate -1
   - 👍 **Like** - Rate 1
   - ⭐ **SuperLike** - Rate 2
4. **Navigate Pages** - Use Previous/Next buttons to browse 50 movies per page
5. **Get Recommendations** - Click the **✨ Get Recommendations** button to:
   - Train an ML model on your rated movies
   - Generate personalized recommendations (top 20)
   - View results in a modal with scores
6. **Track Ratings** - The counter shows total ratings submitted in current session

## API Endpoints

### GET /api/movies
Get paginated and filtered movies.

**Query Parameters:**
- `page` (int, default=1) - Page number
- `limit` (int, default=50) - Items per page
- `search` (str, optional) - Partial title search query

**Response:**
```json
{
  "movies": [
    {"id": 1, "title": "The Shawshank Redemption"},
    {"id": 2, "title": "The Godfather"}
  ],
  "page": 1,
  "limit": 50,
  "total": 143909,
  "total_pages": 2879
}
```

### POST /api/rate
Submit a movie rating.

**Request Body:**
```json
{
  "user_id": 1,
  "movie_id": 550,
  "rating": 1
}
```

**Response:**
```json
{
  "success": true,
  "message": "Rating submitted for movie 550",
  "rating_count": 42
}
```

### GET /api/ratings
Retrieve all ratings submitted in current session.

**Response:**
```json
{
  "ratings": [
    {"user_id": 1, "movie_id": 550, "rating": 1},
    {"user_id": 1, "movie_id": 278, "rating": 2}
  ],
  "count": 2
}
```

### POST /api/recommend
Get AI-powered movie recommendations based on rated movies.

**Request Body:**
```json
{
  "user_id": 1,
  "top_k": 20
}
```

**Response:**
```json
{
  "recommendations": [
    {"id": 12345, "title": "Great Movie", "score": 0.95},
    {"id": 67890, "title": "Another Great Film", "score": 0.88}
  ],
  "count": 20,
  "user_id": 1
}
```

**Note:** This endpoint trains a machine learning model (Two-Tower Neural Network) on your ratings and existing dataset ratings. Training takes 10-30 seconds depending on your hardware. The model uses movie keywords/tags to find semantically similar movies to what you like.

## Accessing Ratings from Python

To access the ratings DataFrame from other modules:

```python
from app import get_ratings_df

# Get the DataFrame
ratings_df = get_ratings_df()

# Use in recommender or other modules
print(ratings_df)
```

## File Structure

```
movieselector3000/
├── app.py                    # Flask application
├── datahandler.py           # Data loading utility
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html          # HTML template
└── static/
    ├── style.css           # CSS styling
    └── script.js           # JavaScript (search, pagination, ratings)
```

## Configuration

Modify these in `app.py` if needed:

```python
# Data file paths
MOVIES_FILE = 'dataset/movies_stripped.csv'
RATINGS_FILE = 'dataset/interactions.csv'
TITLES_FILE = 'dataset/TMDB_movie_dataset_v11.csv'

# Flask port
app.run(debug=True, port=5000)
```

## Notes

- Ratings are stored in-memory during the session
- The `/api/ratings` endpoint shows all ratings submitted so far
- Search is case-insensitive and supports partial title matching
- Default user_id is 1 (change in JavaScript if needed)
- All ratings are accessible to other Python modules via `get_ratings_df()`

## Troubleshooting

**"File not found" errors:**
- Ensure all CSV files exist in the `dataset/` directory
- Check file paths in `app.py`

**Port 5000 already in use:**
- Change the port: `app.run(port=5001)`

**Movies not showing up:**
- Check that `TMDB_movie_dataset_v11.csv` has `id` and `title` columns
- Verify the merge logic in `get_all_movies_with_titles()`

## Future Enhancements

- Persistent database (SQLite/PostgreSQL)
- User authentication and per-user sessions
- Export ratings to CSV/database
- Advanced filtering and sorting
- Real-time recommendation updates
- Batch recommendation generation
- Model caching to speed up repeated recommendations
