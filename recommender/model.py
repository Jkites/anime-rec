import pandas as pd
import pickle
import joblib
from hybrid import hybrid_recommend_popularity 
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"

RATING_CSV = "data/updated_ratings.csv"
ANIME_CSV = "data/updated_animes.csv"
DATA_DIR = "data/processed2"
os.makedirs(DATA_DIR, exist_ok=True)

# cf_model, content_sim, anime_index, idx_to_animeID
CF_MODEL_FILE = os.path.join(DATA_DIR, "cf_model.pkl")
SIM_MATRIX_FILE = os.path.join(DATA_DIR, "content_sim.npz")
A2IDX_FILE = os.path.join(DATA_DIR, "anime_index.pkl")
# IDX2A_FILE = os.path.join(DATA_DIR, "idx_to_animeID.pkl")
BEST_ALPHA_FILE = os.path.join(DATA_DIR, "best_alpha.txt")
from scipy.sparse import load_npz

# load files from cache
ratings = pd.read_csv(RATING_CSV)
anime_meta = pd.read_csv(ANIME_CSV)

svd_model = joblib.load(CF_MODEL_FILE)
content_sim = load_npz(SIM_MATRIX_FILE)
with open(A2IDX_FILE, "rb") as f:
    anime_index = pickle.load(f)
with open(BEST_ALPHA_FILE, "r") as f:
    best_alpha = float(f.read().strip())

anime_avg_rating = ratings.groupby('animeID')['rating'].mean().to_dict()
anime_count = ratings.groupby('animeID')['rating'].count().to_dict()
# !!! convert the anime IDS here to match dataset OR change the anime dataset IDS to match mal.
# probably less computation to convert here then back. but theres no way to query that information. We must change animes.csv
def get_recommendations(user_input, user_id, top_n=10, alpha=0.7):
    # ranked = hybrid_recommend(
    #     user_watched=user_input,
    #     userID=user_id,
    #     anime_meta=anime_meta,
    #     cf_model=svd_model,
    #     content_sim=content_sim,
    #     anime_index=anime_index,
    #     alpha=best_alpha
    # )
    ranked = hybrid_recommend_popularity(
        anime_avg_rating=anime_avg_rating,
        anime_count=anime_count,
        user_watched=user_input,
        userID=user_id,
        anime_meta=anime_meta,
        cf_model=svd_model,
        content_sim=content_sim,
        anime_index=anime_index,
        alpha=best_alpha
    )
    # ranked = user_input
    results = []
    for anime_id, score in ranked:
        match = anime_meta[anime_meta["animeID"] == anime_id]
        if not match.empty:
            row = match.iloc[0]
        else:
            continue
        row = anime_meta[anime_meta["animeID"] == anime_id].iloc[0]
        results.append({
            "title": row["title"],
            "score": round(score, 2),
            "image_url": row["image_url"] or "https://cdn.myanimelist.net/s/common/uploaded_files/1455542152-1164a6a65b3efde6f0d5be12cf67edfc.png",
            "url": row["mal_url"] or "https://myanimelist.net/",
            "global_score": row["score"]
        })
    return results
