#!/bin/bash
# Test script for Flask Movie Rater with Recommendations

echo "=== MovieSelector3000 Flask Setup Test ==="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Load conda module
module load python3/anaconda-2024.07

# Check dependencies
echo "1. Checking dependencies..."
conda run -n torch_env python -c "
import flask, flask_cors, pandas, torch
print('✓ Flask:', flask.__version__)
print('✓ Flask-CORS:', flask_cors.__version__)
print('✓ Pandas version available')
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
" || exit 1

echo ""
echo "2. Checking data files..."
if [ -f "../dataset/movies_stripped.csv" ]; then
    echo "✓ movies_stripped.csv found"
else
    echo "✗ movies_stripped.csv not found"
    exit 1
fi

if [ -f "../dataset/interactions.csv" ]; then
    echo "✓ interactions.csv found"
else
    echo "✗ interactions.csv not found"
    exit 1
fi

if [ -f "../dataset/TMDB_movie_dataset_v11.csv" ]; then
    echo "✓ TMDB_movie_dataset_v11.csv found"
else
    echo "✗ TMDB_movie_dataset_v11.csv not found"
    exit 1
fi

echo ""
echo "3. Checking Flask app syntax..."
conda run -n torch_env python -m py_compile app.py datahandler.py || exit 1
echo "✓ Syntax check passed"

echo ""
echo "4. Testing data loading..."
conda run -n torch_env python -c "
from datahandler import DataHandler
import os

data_dir = '../dataset'
dh = DataHandler(
    os.path.join(data_dir, 'movies_stripped.csv'),
    os.path.join(data_dir, 'interactions.csv'),
    os.path.join(data_dir, 'TMDB_movie_dataset_v11.csv')
)
print('✓ Movies loaded:', len(dh.getAllMovies()))
print('✓ Ratings loaded:', len(dh.getUserRatings()))
print('✓ Titles loaded:', len(dh.getMovieTitles()))
" || exit 1

echo ""
echo "=== All checks passed! ✓ ==="
echo ""
echo "To start the Flask app, run:"
echo "  bash -c 'cd \"$(pwd)\" && module load python3/anaconda-2024.07 && conda run -n torch_env python app.py'"
echo ""
echo "Then open http://localhost:5000 in your browser"
