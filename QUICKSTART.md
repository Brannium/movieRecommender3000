# Quick Start Guide

## Setup

1. **Navigate to project directory:**
   ```bash
   cd ~/Documents/Advanced\ Artificial\ Intelligence/movie-recommender/movieselector3000
   ```

2. **Run setup verification:**
   ```bash
   bash test_setup.sh
   ```

3. **Install dependencies (if needed):**
   ```bash
   bash -c "module load python3/anaconda-2024.07 && conda run -n torch_env pip install flask flask-cors pandas torch"
   ```

## Running the Application

```bash
bash -c "module load python3/anaconda-2024.07 && conda run -n torch_env python app.py"
```

The app will start on `http://localhost:5000`

## Using the App

### Workflow

1. **Rate Movies**
   - Browse through the movie list (50 movies per page)
   - Use the search bar to find specific movies
   - Click rating buttons: 👎 Dislike, 👍 Like, ⭐ SuperLike
   - Watch the rating counter increase

2. **Get Recommendations**
   - Rate at least a few movies
   - Click **✨ Get Recommendations** button
   - Wait for the model to train (10-30 seconds)
   - View personalized recommendations with scores

3. **Manage Your Ratings**
   - Click on a different rating button to change your vote
   - Search to find and re-rate movies
   - All ratings are tracked in the counter

## API Endpoints

### Movies
```
GET /api/movies?page=1&limit=50&search=inception
```

### Submit Rating
```
POST /api/rate
{"user_id": 1, "movie_id": 550, "rating": 1}
```

### Get Ratings
```
GET /api/ratings
```

### Get Recommendations
```
POST /api/recommend
{"user_id": 1, "top_k": 20}
```

## Troubleshooting

### "Port 5000 already in use"
- Kill existing Flask process: `pkill -f "python app.py"`
- Or change port in `app.py`: `app.run(debug=True, port=5001)`

### "No recommendations available"
- Ensure you've rated at least a few movies before clicking recommendations

### Data loading errors
- Verify CSV files exist in `../dataset/`
- Check file paths in `app.py`

### Model training takes too long
- This is normal (10-30 seconds on CPU, faster on GPU)
- GPU is available (CUDA enabled)

## Architecture

```
app.py              - Flask backend with API endpoints
datahandler.py      - Data loading and preprocessing
recommender.py      - ML model for recommendations
templates/
  └─ index.html     - HTML frontend
static/
  ├─ style.css      - Styling
  └─ script.js      - Client-side JavaScript
```

## Key Technologies

- **Flask** - Web framework
- **Pandas** - Data manipulation
- **PyTorch** - Deep learning (Two-Tower Neural Network)
- **Gensim** - Word embeddings
- **JavaScript** - Interactive frontend
