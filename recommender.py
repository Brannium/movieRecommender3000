# purpose
# - implement recommendation algorithms
# - interface with datahandler for user ratings and movie data
import pandas as pd
import numpy as np
from ast import literal_eval

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from gensim.models import Word2Vec

MAX_USER_MOVIES = 20  # max number of positively rated movies to consider per user
MAX_TAGS_PER_MOVIE = 64  # max number of tags to consider per movie
PAD_IDX = 0  # padding index used by embeddings


class MovieRecommender():

    def __init__(self, allMovies: pd.DataFrame, userRatings: pd.DataFrame, movieTitles: pd.DataFrame, device: torch.device):
        self.allMovies = allMovies
        self.userRatings = userRatings
        self.movieTitles = movieTitles
        self.device = device
        self.model = None

    def _rating_to_label_strength(self, r):
        if r == -1:
            return 0.0, 1.0      # explicit negative, normal weight
        elif r == 1:
            return 1.0, 1.0      # like, normal weight
        elif r == 2:
            return 1.0, 1.5      # big like, higher weight
        else:  # 0 or anything else
            return None, None

    def _tags_to_ids(self, tag_list, tag2idx, max_tags=MAX_TAGS_PER_MOVIE):
        ids = [tag2idx.get(t, 0) for t in tag_list][:max_tags]
        if len(ids) < max_tags:
            ids += [0] * (max_tags - len(ids))
        return ids

    def _build_user_tag_bag_for_user(self, user_id, movieid2tags, max_user_movies=MAX_USER_MOVIES):
        userRatings = self.userRatings
        pos_movies = (
            userRatings[
                (userRatings["user_id"] == user_id) &
                (userRatings["label"] == 1.0)
            ]["movie_id"]
            .tolist()
        )

        if not pos_movies:
            return np.zeros(MAX_TAGS_PER_MOVIE * max_user_movies, dtype=np.int64)

        if len(pos_movies) > max_user_movies:
            pos_movies = np.random.choice(pos_movies, max_user_movies, replace=False)

        tag_ids = []
        for mid in pos_movies:
            tag_ids.extend(movieid2tags.get(mid, [0]*MAX_TAGS_PER_MOVIE))

        max_len = MAX_TAGS_PER_MOVIE * max_user_movies
        tag_ids = tag_ids[:max_len]
        if len(tag_ids) < max_len:
            tag_ids += [0] * (max_len - len(tag_ids))

        return np.array(tag_ids, dtype=np.int64)

    def recommend_for_user(self, user_id, movieid2tags, item_vecs, all_movie_ids, top_k=20, exclude_seen=True):
        user_tag_ids = self._build_user_tag_bag_for_user(user_id, movieid2tags)
        user_tag_ids = torch.tensor(user_tag_ids, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            u_vec = self.model.user_encoder(user_tag_ids)           # [1, D]
            item_vecs_device = item_vecs.to(self.device)            # Move item_vecs to device
            scores = (u_vec @ item_vecs_device.t()).squeeze(0) # [N_movies]

        scores_np = scores.cpu().numpy()
        ranked_idx = np.argsort(-scores_np)

        if exclude_seen:
            seen_movies = set(
                self.userRatings[self.userRatings["user_id"] == user_id]["movie_id"].tolist()
            )
            ranked_idx = [i for i in ranked_idx if all_movie_ids[i] not in seen_movies]

        top_idx = ranked_idx[:top_k]
        rec_ids = [all_movie_ids[i] for i in top_idx]
        rec_scores = scores_np[top_idx]

        return list(zip(rec_ids, rec_scores))

    def train_recommender(self):
        labels = []
        weights = []
        for r in self.userRatings["rating"]:
            y, w = self._rating_to_label_strength(r)
            labels.append(y)
            weights.append(w)

        self.userRatings["label"] = labels
        self.userRatings["sample_weight"] = weights

        # Drop rows where label is None (rating == 0)
        self.userRatings = self.userRatings[self.userRatings["label"].notnull()].reset_index(drop=True)
        self.userRatings["label"] = self.userRatings["label"].astype(np.float32)
        self.userRatings["sample_weight"] = self.userRatings["sample_weight"].astype(np.float32)

        # Build tag corpus: each movie's tags is one "sentence"
        tag_corpus = movies_df["tags"].tolist()
        print(f"Tag corpus sample: {tag_corpus[0]}")  # print first 5 for inspection

        # Train Word2Vec on tag co-occurrence in movies
        w2v_dim = 512
        w2v_model = Word2Vec(
            sentences=tag_corpus,
            vector_size=w2v_dim,
            window=5,
            min_count=1,   # keep all tags; raise to drop very rare ones
            sg=1,          # skip-gram
            workers=4
        )

        # Build tag vocabulary
        all_tags = sorted({t for tags in tag_corpus for t in tags})
        print(f"Total unique tags: {len(all_tags)}, sample: {all_tags[400:430]}")
        tag2idx = {tag: i + 1 for i, tag in enumerate(all_tags)}  # 0 reserved for PAD

        vocab_size = len(tag2idx) + 1
        emb_dim = w2v_dim

        # Initialize embedding matrix with Word2Vec vectors
        emb_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
        for tag, idx in tag2idx.items():
            if tag in w2v_model.wv:
                emb_matrix[idx] = w2v_model.wv[tag]
            else:
                emb_matrix[idx] = np.random.normal(scale=0.01, size=(emb_dim,))

        MAX_TAGS_PER_MOVIE = 64  # adjust

        movies_df["tag_ids"] = movies_df["tags"].apply(lambda tags: self._tags_to_ids(tags, tag2idx))

        # lookup: movie_id -> tag_id list
        movieid2tags = dict(zip(movies_df["id"], movies_df["tag_ids"]))
        all_movie_ids = movies_df["id"].tolist()

        # user_id -> list of positively rated movie_ids
        user_pos_movies = (
            self.userRatings[self.userRatings["label"] == 1.0]
            .groupby("user_id")["movie_id"]
            .apply(list)
            .to_dict()
        )

        dataset = InteractionDataset(self.userRatings, movieid2tags, user_pos_movies)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)

        PAD_IDX = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TwoTowerRec(vocab_size, emb_dim, emb_matrix).to(device)
        criterion = nn.BCEWithLogitsLoss(reduction="none")   # so we can apply weights
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        EPOCHS = 100

        for epoch in range(EPOCHS):
            self.model.train()
            epoch_loss = 0.0
            n_samples = 0

            for user_tags, movie_tags, labels, weights in dataloader:
                user_tags = user_tags.to(device)
                movie_tags = movie_tags.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                optimizer.zero_grad()
                logits = self.model(user_tags, movie_tags)              # [B]
                loss_vec = criterion(logits, labels)               # [B]
                # weight big likes more
                loss = (loss_vec * weights).mean()

                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                n_samples += batch_size

            print(f"Epoch {epoch+1}/{EPOCHS} - loss: {epoch_loss / n_samples:.4f}")

        self.model.eval()

        batch_size_inference = 512  # Process movies in batches

        all_item_vecs = []
        all_movie_ids = movies_df["id"].tolist()

        with torch.no_grad():
            for i in range(0, len(movies_df), batch_size_inference):
                batch_tag_ids = movies_df.iloc[i:i+batch_size_inference]["tag_ids"].values
                batch_tensor = torch.tensor(
                    np.stack(batch_tag_ids),
                    dtype=torch.long
                ).to(device)
                
                batch_vecs = self.model.item_encoder(batch_tensor)
                all_item_vecs.append(batch_vecs.cpu())
                del batch_tensor
                torch.cuda.empty_cache()

        item_vecs = torch.cat(all_item_vecs, dim=0)
        item_vecs_np = item_vecs.cpu().numpy()


        # Example:
        user_id_example = 1
        recommendations = self.recommend_for_user(user_id_example, movieid2tags, item_vecs, all_movie_ids, top_k=30)
        
        for movie_id, score in recommendations:
            movie_row = self.movieTitles[self.movieTitles["id"] == movie_id]
            if not movie_row.empty:
                title = movie_row.iloc[0]["title"]
                print(f"Movie ID: {movie_id}, Title: {title}, Score: {score:.4f}")
            else:
                print(f"Movie ID: {movie_id}, Title: Unknown, Score: {score:.4f}")

        # print out the movies watched by the user with their ratings, formatted as table
        watched_movies = self.userRatings[self.userRatings["user_id"] == user_id_example]
        watched_movies = watched_movies.merge(self.movieTitles, left_on="movie_id", right_on="id", how="left")
        watched_movies = watched_movies[["title", "rating"]]
        print("\nMovies watched by user:")
        print(watched_movies.to_string(index=False))

        # print a list of recommended movies (titles with ratings)
        print("\nRecommended movies:")
        for movie_id, score in recommendations:
            movie_row = self.movieTitles[self.movieTitles["id"] == movie_id]
            if not movie_row.empty:
                title = movie_row.iloc[0]["title"]
                print(f"{score:.6f} - {title}")
            else:
                print(f"- Unknown (ID: {movie_id})")
    

class InteractionDataset(Dataset):
    def __init__(self, df, movieid2tags, user_pos_movies,
                max_user_movies=MAX_USER_MOVIES, max_tags_per_movie=MAX_TAGS_PER_MOVIE):
        self.df = df.reset_index(drop=True)
        self.movieid2tags = movieid2tags
        self.user_pos_movies = user_pos_movies
        self.max_user_movies = max_user_movies
        self.max_tags_per_movie = max_tags_per_movie
    def _user_tag_bag(self, user_id, target_movie_id):
        # Movies this user liked (not including the target)
        pos_movies = self.user_pos_movies.get(user_id, [])
        pos_movies = [m for m in pos_movies if m != target_movie_id]

        if not pos_movies:
            # cold user: all PAD
            return [0] * (self.max_tags_per_movie * self.max_user_movies)

        if len(pos_movies) > self.max_user_movies:
            pos_movies = np.random.choice(pos_movies, self.max_user_movies, replace=False)

        tag_ids = []
        for mid in pos_movies:
            tag_ids.extend(self.movieid2tags.get(mid, [0]*self.max_tags_per_movie))

        max_len = self.max_tags_per_movie * self.max_user_movies
        tag_ids = tag_ids[:max_len]
        if len(tag_ids) < max_len:
            tag_ids += [0] * (max_len - len(tag_ids))

        return tag_ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = row["user_id"]
        movie_id = row["movie_id"]
        label = row["label"]
        weight = row["sample_weight"]

        user_tag_ids = self._user_tag_bag(user_id, movie_id)  # [U]
        movie_tag_ids = self.movieid2tags.get(movie_id, [0]*MAX_TAGS_PER_MOVIE)

        return (
            torch.tensor(user_tag_ids, dtype=torch.long),
            torch.tensor(movie_tag_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )

class TagEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, emb_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)

        if emb_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
            # If you want to freeze pretrained vectors at the start:
            # self.embedding.weight.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, tag_ids):
        """
        tag_ids:
        - user:  [B, U]
        - movie: [B, T]
        """
        emb = self.embedding(tag_ids)                      # [B, L, D]
        mask = (tag_ids != PAD_IDX).float().unsqueeze(-1)  # [B, L, 1]
        summed = (emb * mask).sum(dim=1)                   # [B, D]
        counts = mask.sum(dim=1).clamp_min(1.0)            # [B, 1]
        pooled = summed / counts                           # mean pooling
        pooled = self.proj(pooled)
        return F.normalize(pooled, dim=-1)                 # [B, D]


class TwoTowerRec(nn.Module):
    def __init__(self, vocab_size, emb_dim, emb_matrix=None):
        super().__init__()
        self.user_encoder = TagEncoder(vocab_size, emb_dim, emb_matrix)
        self.item_encoder = TagEncoder(vocab_size, emb_dim, emb_matrix)

    def forward(self, user_tag_ids, movie_tag_ids):
        """
        returns logits (un-sigmoid-ed scores)
        """
        u_vec = self.user_encoder(user_tag_ids)    # [B, D]
        i_vec = self.item_encoder(movie_tag_ids)   # [B, D]

        logits = (u_vec * i_vec).sum(dim=-1)       # [B]
        logits = logits / 0.2                      # temperature scaling
        return logits



# ---------------------------------------------------
# Load movies
# ---------------------------------------------------
movies_df = pd.read_csv("data/movies.csv")  # must contain 'id' and 'tags'

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
interactions_df = pd.read_csv("data/interactions.csv")
# expected columns: user_id, movie_id, rating

# load full movies dataset for title lookup
movie_titles_df = pd.read_csv("../dataset/TMDB_movie_dataset_v11.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
movieRecommender = MovieRecommender(allMovies=movies_df, userRatings=interactions_df, movieTitles=movie_titles_df, device=device)
movieRecommender.train_recommender()