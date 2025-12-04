# provide
# - load movie database
# - save and load user ratings
import pandas as pd
import json

class DataHandler():
    def __init__(self, movies_filepath, ratings_filepath, movie_titles_filepath):
        self.movies_df = pd.read_csv(movies_filepath)                # must contain 'id' and 'tags'
        self.interactions_df = pd.read_csv(ratings_filepath)         # expected columns: user_id, movie_id, rating
        self.movie_titles_df = pd.read_csv(movie_titles_filepath)    # load full movies dataset for title lookup

        # convert tags to list of strings that are divided by comma
        if isinstance(self.movies_df.loc[0, "tags"], str):
            self.movies_df["tags"] = self.movies_df["tags"].apply(lambda x: [tag.strip() for tag in x.split(",")])

        print(f"Loaded {len(self.movies_df)} movies.")
        
    def getAllMovies(self):
        return self.movies_df

    def getUserRatings(self):
        return self.interactions_df
    
    def getMovieTitles(self):
        return self.movie_titles_df

    def getMovieTitle(self, movie_id, default="Unknown"):
        movie_row = self.movie_titles_df[self.movie_titles_df["id"] == movie_id]
        if not movie_row.empty:
            return movie_row.iloc[0]["title"]
        else:
            return default

    def notusedyet():
        # print out the movies watched by the user with their ratings, formatted as table
        watched_movies = interactions_df[interactions_df["user_id"] == 1]
        watched_movies = watched_movies.merge(movie_titles_df, left_on="movie_id", right_on="id", how="left")
        watched_movies = watched_movies[["title", "rating"]]
        print("\nMovies watched by user:")
        print(watched_movies.to_string(index=False))

        # print a list of recommended movies (titles with ratings)
        print("\nRecommended movies:")
        for movie_id, score in recommendations:
            movie_row = movie_titles_df[movie_titles_df["id"] == movie_id]
            if not movie_row.empty:
                title = movie_row.iloc[0]["title"]
                print(f"{score:.6f} - {title}")
            else:
                print(f"- Unknown (ID: {movie_id})")