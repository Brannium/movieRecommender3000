# purpose
# - run the main Streamlit app logic
# - integrate datahandler and recommender modules
import pandas as pd
import torch

from recommender import MovieRecommender


if __name__ == "__main__":

    # ---------------------------------------------------
    # Load datasets
    # ---------------------------------------------------
    movies_df = pd.read_csv("data/movies.csv")  # must contain 'id' and 'tags'
    interactions_df = pd.read_csv("data/interactions.csv") # expected columns: user_id, movie_id, rating
    movie_titles_df = pd.read_csv("../dataset/TMDB_movie_dataset_v11.csv") # load full movies dataset for title lookup

    # If tags are stored as string like "['space', 'alien']"
    #if isinstance(movies_df.loc[0, "tags"], str):
    #    movies_df["tags"] = movies_df["tags"].apply(literal_eval)

    print(f"Loaded {len(movies_df)} movies.")

    # convert tags to list of strings that are divided by comma
    if isinstance(movies_df.loc[0, "tags"], str):
        movies_df["tags"] = movies_df["tags"].apply(lambda x: [tag.strip() for tag in x.split(",")])
    print(movies_df.head())
    print(movies_df["tags"].dtypes)

    # ---------------------------------------------------
    # Load interactions
    # ---------------------------------------------------


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    movieRecommender = MovieRecommender(allMovies=movies_df, userRatings=interactions_df, movieTitles=movie_titles_df, device=device)
    movieRecommender.train_recommender()