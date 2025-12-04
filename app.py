# purpose
# - run the main Streamlit app logic
# - integrate datahandler and recommender modules
import pandas as pd
import torch

from recommender import MovieRecommender
from datahandler import DataHandler
from ui import MovieSelector3000UI


if __name__ == "__main__":

    # Load data
    data_handler = DataHandler(
        movies_filepath="data/movies.csv",
        ratings_filepath="data/interactions.csv",
        movie_titles_filepath="../dataset/TMDB_movie_dataset_v11.csv"
    )

    movieUI = MovieSelector3000UI()
    movieIDTitlePairs = data_handler.getMovieTitles().apply(lambda row: (row['id'], row['title']), axis=1).tolist()
    movieUI.render_rating_screen(movieIDTitlePairs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    movieRecommender = MovieRecommender(allMovies=data_handler.getAllMovies(), userRatings=data_handler.getUserRatings(), movieTitles=data_handler.getMovieTitles(), device=device)
    movieRecommender.train_recommender()

    recommendations = movieRecommender.recommend_for_user(user_id=1)
    
    print("\nTop recommended movies for user 1:")
    for movie_id, score in recommendations:
        title = data_handler.getMovieTitle(movie_id)
        print(f"Score: {score:.7f}, Title: {title}")
