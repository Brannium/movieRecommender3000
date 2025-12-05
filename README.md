# MovieSelector3000 - Flask Movie Rating UI

## Setup

### 1. Install Dependencies

```bash
cd movie-recommender/movieselector3000
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure these files exist in the `dataset/` directory:
- `movies_stripped.csv` - Contains movie IDs
- `interactions.csv` - User ratings (columns: user_id, movie_id, rating), currently not used but need to be present
- `TMDB_movie_dataset_v11.csv` - Movie titles and metadata ([download here](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies))

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

1. **Open the Web UI** - Navigate to `http://localhost:5000` in your browser
2. **Search Movies** - Type in the search box to find movies by title (partial matching supported)
3. **Rate Movies** - Click buttons next to each movie:
   - 👎 **Dislike**
   - 👍 **Like**
   - ⭐ **SuperLike**
4. **Navigate Pages** - Use Previous/Next buttons to browse 50 movies per page
5. **Get Recommendations** - Click the **✨ Get Recommendations** button to:
   - Train an ML model on your rated movies
   - Generate personalized recommendations (top 50)
   - View results in a modal with scores
6. **Track Ratings** - The counter shows total ratings submitted in current session


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
RATINGS_FILE = 'dataset/interactions.csv' # not actually used
TITLES_FILE = 'dataset/TMDB_movie_dataset_v11.csv'

# Flask port
app.run(debug=True, port=5000)
```

## Notes

- Ratings are stored client-side in-memory during the session
- Search is case-insensitive and supports partial title matching

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
