import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
import pandas as pd
import joblib
import pickle
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

RATING_CSV = "data/ratings.csv"
ANIME_CSV = "data/animes.csv"
DATA_DIR = "data/processed2"
os.makedirs(DATA_DIR, exist_ok=True)

# cf_model, content_sim, anime_index, idx_to_animeID
CF_MODEL_FILE = os.path.join(DATA_DIR, "cf_model.pkl")
# TFIDF_FILE = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
SIM_MATRIX_FILE = os.path.join(DATA_DIR, "content_sim.npy")
A2IDX_FILE = os.path.join(DATA_DIR, "anime_index.pkl")
IDX2A_FILE = os.path.join(DATA_DIR, "idx_to_animeID.pkl")

LOAD_ONLY = False  # set True to skip retraining if cache exists

# cap ratings for now
MAX_RATINGS = 2_000_000

def load():
    print("reading ratings csv...")
    ratings = pd.read_csv(RATING_CSV)  # (userID, animeID, rating)
    print("reading anime csv...")
    anime_meta = pd.read_csv(ANIME_CSV)  # (animeID, title, alternative_title, type, year, score, episodes, mal_url, sequel, image_url, genres, genres_detailed)

    # oguri cap
    if len(ratings) > MAX_RATINGS:
        ratings = ratings.sample(MAX_RATINGS, random_state=1)

    # split into train test
    reader = Reader(rating_scale=(0.1, 10.0))
    print("loading dataset from df (capped)")
    data = Dataset.load_from_df(ratings[['userID', 'animeID', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=1)
    return ratings, anime_meta, trainset, testset

# train SVD model
def trainSVD(trainset, testset):
    print("training SVD model")
    cf_model = SVD(n_factors=100, reg_all=0.02, random_state=42, verbose = True)
    cf_model.fit(trainset)

    # Evaluate CF alone
    cf_preds = cf_model.test(testset)
    cf_rmse = accuracy.rmse(cf_preds, verbose=True)
    print(f"SVD Test RMSE: {cf_rmse:.4f}") # 1.6828

    return cf_model

def trainContentBased(anime_meta):
    # combine genres and synopsis into one text field
    print("combining features into one field")
    # features = ['genres','genres_detailed ','type','title', 'alternative_title']
    # anime_meta['text_features'] = (
    #     anime_meta['genre'].fillna('') + ' ' + anime_meta['synopsis'].fillna('')
    # )
    anime_meta['text_features'] = (
        anime_meta['genres'].fillna('') + 
        ' ' + anime_meta['genres_detailed'].fillna('') + 
        ' ' + anime_meta['type'].fillna('') + 
        ' ' + anime_meta['title'].fillna('') +
        ' ' + anime_meta['alternative_title'].fillna('')
    )

    # vectorize text
    print("vectorizing text...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(anime_meta['text_features'])

    # compute similarity matrix (item-item)
    print("computing ismilarity matrix...")
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # map animeID to index in similarity matrix
    print("mapping ids to index")
    anime_index = {aid: idx for idx, aid in enumerate(anime_meta['animeID'])}
    idx_to_animeID = {idx: aid for aid, idx in anime_index.items()}
    return content_sim, anime_index, idx_to_animeID

# # average similarity of candidate to items the user has watched
# def content_based_score(user_watched_ids, candidate_id, content_sim, anime_index):
#     if candidate_id not in anime_index:
#         return 0
#     cand_idx = anime_index[candidate_id]
#     sims = []
#     for wid in user_watched_ids:
#         if wid in anime_index:
#             sims.append(content_sim[cand_idx, anime_index[wid]])
#     return np.mean(sims) if sims else 0

def hybrid_rmse(testset, anime_index, content_sim, cf_items_seen, cf_model, alpha=0.7):
    y_true, y_pred = [], []
    for uid, iid, true_r in testset:
        cb_score = 0
        if iid in anime_index:
            idx = anime_index[iid]
            cb_score = np.mean(content_sim[idx])  # average similarity score

        if iid in cf_items_seen:
            cf_score = cf_model.predict(uid, iid).est
            pred = alpha * cf_score + (1 - alpha) * cb_score
        else:
            pred = cb_score  # CB fallback

        y_true.append(true_r)
        y_pred.append(pred)

    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)) #2.8163 :(

def hybrid_recommend(userID, ratings, anime_meta, cf_model, content_sim, anime_index, idx_to_animeID, top_n=10, alpha=0.7):
    # blend CF and CB recommendations with alpha
    # get all anime IDs
    # all_animeIDs = anime_meta['animeID'].tolist()

    # get watched anime for the user
    print("getting watched animes...")
    user_watched = ratings[ratings['userID'] == userID]['animeID'].tolist()
    # watched_ids = user_watched['animeID'].tolist()

    # sum similarities for watched anime
    print("summing similarity scores")
    sim_vector = np.sum(content_sim[[anime_index[aid] for aid in user_watched if aid in anime_index]], axis=0)
    sim_scores = {idx_to_animeID[i]: sim_vector[i] for i in range(len(sim_vector))}

    # unique seen anime IDs by CF
    cf_items_seen = set(ratings['animeID'].unique())

    final_scores = {}
    print("scoring unseen anime")
    for animeID in anime_meta['animeID']:
        if animeID in user_watched:
            continue
        cb_score = sim_scores.get(animeID, 0)
        if animeID in cf_items_seen: # blend
            cf_score = cf_model.predict(userID, animeID).est
            final_scores[animeID] = alpha * cf_score + (1 - alpha) * cb_score
        else: # just cb
            final_scores[animeID] = cb_score

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    hybrid_rmse_val = hybrid_rmse(testset, anime_index, content_sim, cf_items_seen, cf_model, alpha=0.7)
    print(f"Hybrid RMSE: {hybrid_rmse_val:.4f}")

    return ranked[:top_n]
    # scores = []
    # for aid in all_animeIDs:
    #     # skip items the user already watched
    #     if aid in user_watched:
    #         continue

    #     # try SVD prediction
    #     try:
    #         cf_score = cf_model.predict(userID, aid).est
    #     except:
    #         cf_score = None

    #     # Content based score
    #     cb_score = content_based_score(user_watched, aid, content_sim, anime_index)

    #     # blend otherwise just cb
    #     if cf_score is not None and not np.isnan(cf_score):
    #         final_score = alpha * cf_score + (1 - alpha) * cb_score
    #     else:
    #         final_score = cb_score

    #     scores.append((aid, final_score))

    # # return top_n
    # scores.sort(key=lambda x: x[1], reverse=True)
    # return scores[:top_n]
    
if __name__ == "__main__":
    if LOAD_ONLY and all(os.path.exists(f) for f in [CF_MODEL_FILE, SIM_MATRIX_FILE, A2IDX_FILE, IDX2A_FILE]):
        print("Loading recommender from cache...")
        svd_model = joblib.load(CF_MODEL_FILE)
        # tfidf = joblib.load(TFIDF_FILE)
        content_sim = np.load(SIM_MATRIX_FILE)
        with open(A2IDX_FILE, "rb") as f:
            anime_index = pickle.load(f)
        with open(IDX2A_FILE, "rb") as f:
            idx_to_anime_id = pickle.load(f)
        ratings, anime_meta, trainset, testset = load()
    else:
        ratings, anime_meta, trainset, testset = load()

        svd_model = trainSVD(trainset=trainset, testset=testset)

        content_sim, anime_index, idx_to_animeID = trainContentBased(anime_meta)

        # save
        joblib.dump(svd_model, CF_MODEL_FILE)
        #
        np.save(SIM_MATRIX_FILE, content_sim)
        with open(A2IDX_FILE, "wb") as f:
            pickle.dump(anime_index, f)
        with open(IDX2A_FILE, "wb") as f:
            pickle.dump(idx_to_animeID, f)

    userID = random.choice(ratings['userID'].unique())
    recommendations = hybrid_recommend(userID, ratings, anime_meta, svd_model, content_sim, anime_index, idx_to_animeID, top_n=10)

    # Map to anime titles
    print(f"Top 10 recommendations for user {userID}:")
    for aid, score in recommendations:
        title = anime_meta.loc[anime_meta['animeID'] == aid, 'title'].values[0]
        print(f"{title} (Score: {score:.4f})")

