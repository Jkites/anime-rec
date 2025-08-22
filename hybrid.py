import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
import pandas as pd
import joblib
import pickle
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.model_selection import train_test_split
# from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from collections import defaultdict
from scipy.sparse import save_npz, load_npz
import matplotlib.pyplot as plt

RATING_CSV = "data/updated_ratings.csv"
ANIME_CSV = "data/updated_animes.csv"
DATA_DIR = "data/processed2"
os.makedirs(DATA_DIR, exist_ok=True)

# cf_model, content_sim, anime_index, idx_to_animeID
CF_MODEL_FILE = os.path.join(DATA_DIR, "cf_model.pkl")
# TFIDF_FILE = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
SIM_MATRIX_FILE = os.path.join(DATA_DIR, "content_sim.npz")
A2IDX_FILE = os.path.join(DATA_DIR, "anime_index.pkl")
IDX2A_FILE = os.path.join(DATA_DIR, "idx_to_animeID.pkl")
BEST_ALPHA_FILE = os.path.join(DATA_DIR, "best_alpha.txt")

LOAD_ONLY = True  # set True to skip retraining if cache exists

# cap ratings for now
MAX_RATINGS = 4_000_000

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
    # data = Dataset.load_from_df(ratings[['userID', 'animeID', 'rating']], reader)
    full_trainset, testset = train_test_split(ratings, test_size=0.2, random_state=1)
    trainset, valset = train_test_split(full_trainset, test_size=0.1, random_state=1)

    # convert to surprise dataset
    trainset = Dataset.load_from_df(trainset[['userID', 'animeID', 'rating']], reader).build_full_trainset()
    valset = Dataset.load_from_df(valset[['userID', 'animeID', 'rating']], reader).build_full_trainset().build_testset()
    testset = Dataset.load_from_df(testset[['userID', 'animeID', 'rating']], reader).build_full_trainset().build_testset()
    
    return ratings, anime_meta, trainset, valset, testset

# !!!TODO: crossvalidation hyperparameter search / switch to lightFM / periodic retrain?
# train SVD model
def trainSVD(trainset, testset):
    print("training SVD model")
    cf_model = SVD(n_factors=100, reg_all=0.02, random_state=42, verbose = True)
    cf_model.fit(trainset)

    # Evaluate CF alone
    cf_preds = cf_model.test(testset)
    cf_rmse = accuracy.rmse(cf_preds, verbose=True)
    print(f"SVD Test RMSE: {cf_rmse:.4f}") # 1.6828 1.6539

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
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix, dense_output=False)

    # map animeID to index in similarity matrix
    print("mapping ids to index")
    anime_index = {aid: idx for idx, aid in enumerate(anime_meta['animeID'])}
    idx_to_animeID = {idx: aid for aid, idx in anime_index.items()}
    return content_sim, anime_index, idx_to_animeID

# # average similarity of candidate to items the user has watched
# def content_based_score(user_watched_ids, userID, content_sim, anime_index):
#     if userID not in anime_index:
#         return 0
#     cand_idx = anime_index[userID]
#     sims = []
#     for wid in user_watched_ids:
#         if wid in anime_index:
#             sims.append(content_sim[cand_idx, anime_index[wid]])
#     return np.mean(sims) if sims else 0
# weighted average of user's own ratings
def cb_predict(animeID, user_watched, content_sim, anime_index):
    if animeID not in anime_index:
        return None  # not in data
    
    cand_idx = anime_index[animeID]
    # check if exist in similarity matrix
    rated_idxs = []
    ratings = []
    for aid, r in user_watched:
        if aid in anime_index:
            rated_idxs.append(anime_index[aid])
            ratings.append(r)
    
    if not rated_idxs:
        return None  # no overlap
    
    sims = content_sim[cand_idx, rated_idxs].toarray().ravel()  # similarity vector to user's rated items
    ratings = np.array(ratings, dtype=np.float32)

    numerator = np.dot(sims, ratings)
    denominator = np.sum(sims)
    
    if denominator <= 0:
        return None
    
    pred = numerator / denominator
    # clip to valid range
    return float(np.clip(pred, 0.1, 10.0))

# def precompute_cb_scores(testset, anime_index, content_sim):
#     print("precomputing cb scores...")
#     cb_scores = {}
#     for _, iid, _ in testset:
#         if iid in anime_index and iid not in cb_scores:
#             idx = anime_index[iid]
#             row = content_sim[idx].toarray().ravel() # only sparse works
#             cb_scores[iid] = row.mean()
#     return cb_scores

# min max normalziing
def normalize_cb_predictions(cb_scores, target_min=0.1, target_max=10.0):
    if not cb_scores:
        return cb_scores
    cb_min = min(cb_scores.values())
    cb_max = max(cb_scores.values())
    if cb_max == cb_min:
        return {k: target_min for k in cb_scores}
    scale = (target_max - target_min) / (cb_max - cb_min)
    return {
        k: float(np.clip((v - cb_min) * scale + target_min, target_min, target_max))
        for k, v in cb_scores.items()
    }


def hybrid_rmse(testset, anime_index, cf_items_seen, cf_model, alpha=0.7):
    y_true, y_pred = [], []

    user_test_items = defaultdict(list)
    for uid, iid, true_r in testset:
        user_test_items[uid].append((iid, true_r))

    for uid, items in user_test_items.items():
        user_watched = cf_items_seen.get(uid, [])

        # precompute personalized CB predictions for this user's test items
        cb_scores = {}
        for iid, _ in items:
            score = cb_predict(iid, user_watched, content_sim, anime_index)
            if score is not None:
                cb_scores[iid] = score
        cb_scores = normalize_cb_predictions(cb_scores)

        for iid, true_r in items:
            if iid in cb_scores and uid in cf_items_seen:
                cf_score = cf_model.predict(uid, iid).est
                pred = alpha * cf_score + (1 - alpha) * cb_scores[iid]
            elif uid in cf_items_seen:
                pred = cf_model.predict(uid, iid).est
            elif iid in cb_scores:
                pred = cb_scores[iid]
            else:
                pred = 7.0  # fallback in theory should never occur if animes.csv is complete
            y_true.append(true_r)
            y_pred.append(pred)


    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)) #2.8163 :( 2.8236. 2.4519

def find_best_alpha(val_data, cf_model, trainset, content_sim, anime_index):
    alphas = np.linspace(0, 1, 21)
    best_alpha, best_rmse = None, float("inf")
    
    cf_preds = []
    cb_preds = []
    y_true = []
    user_history_map = defaultdict(list)
    for uid, iid, r in trainset.build_testset():
        user_history_map[uid].append((iid, r))
    for uid, iid, r in val_data:
        # user_watched = ratings[ratings['userID'] == uid]['animeID'].tolist()
        y_true.append(r)
        cf_pred = cf_model.predict(uid, iid).est
        cb_pred = cb_predict(iid, user_history_map[uid], content_sim, anime_index)
        cf_preds.append(cf_pred)
        cb_preds.append(cb_pred if cb_pred is not None else cf_pred)
    
    cf_preds = np.array(cf_preds, dtype=np.float32)
    cb_preds = np.array(cb_preds, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)

    # plt.hist(cf_preds, bins=50, alpha=0.5, label='CF')
    # plt.hist(cb_preds, bins=50, alpha=0.5, label='CB')
    # plt.legend()
    # plt.title("Distribution of CF vs CB Predictions")
    # plt.show()

    for a in alphas:
        preds = a * cf_preds + (1 - a) * cb_preds
        rmse = np.sqrt(np.mean((y_true - preds) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = a
    
    return best_alpha, best_rmse # 1.6441

# evaluation method
def hybrid_recommend_rmse(userID, ratings, trainset, anime_meta, cf_model, content_sim, anime_index, top_n=10, alpha=0.7): # removed indx_to_animeID param unused
    # blend CF and CB recommendations with alpha
    # get all anime IDs
    # all_animeIDs = anime_meta['animeID'].tolist()

    # get watched anime for the user
    print("getting watched animes...")
    user_watched = ratings[ratings['userID'] == userID][['animeID', 'rating']].values.tolist() # data "cheating"
                                                        # but this isn't the evaluation so I think its okay
    # user_watched_param = trainset[trainset['userID'] == userID][['animeID', 'rating']].values.tolist()
    # sum similarities for watched anime
    print("summing similarity scores")
    # sim_vector = np.sum(content_sim[[anime_index[aid] for aid in user_watched if aid in anime_index]], axis=0)
    # sim_scores = {idx_to_animeID[i]: sim_vector[i] for i in range(len(sim_vector))}

    # unique seen anime IDs by CF
    # cf_items_seen = ratings.groupby('userID')[['animeID', 'rating']].apply(
    #     lambda df: list(zip(df['animeID'], df['rating']))
    # ).to_dict()

    cf_items_seen = {}
    for uid in trainset.all_users():  # numeric inner IDs
        raw_uid = trainset.to_raw_uid(uid)  # convert to original userID
        user_ratings = []
        for iid, rating in trainset.ur[uid]:  # ur maps inner user ID -> list of (item_inner_id, rating)
            raw_iid = trainset.to_raw_iid(iid)
            user_ratings.append((raw_iid, rating))
        cf_items_seen[raw_uid] = user_ratings
    watched_ids = set(aid for aid, _ in user_watched)
    cb_scores = {}
    print("computing CB predictions...")
    for animeID in anime_meta['animeID']:
        if animeID in watched_ids:
            continue
        score = cb_predict(animeID, user_watched, content_sim, anime_index)
        if score is not None:
            cb_scores[animeID] = score

    # normalize CB predictions to match CF rating range
    cb_scores = normalize_cb_predictions(cb_scores)

    final_scores = {}
    print("scoring unseen anime")
    for animeID in anime_meta['animeID']:
        if animeID in watched_ids:
            continue
        # cb_score = sim_scores.get(animeID, 0)
        # cb_score = cb_predict(animeID, user_watched, content_sim, anime_index)
        cb_score = cb_scores.get(animeID)
        if animeID in cf_items_seen: # blend
            cf_score = cf_model.predict(userID, animeID).est
            final_scores[animeID] = alpha * cf_score + (1 - alpha) * (cb_score if cb_score is not None else cf_score)
        # else: # just cb
        #     final_scores[animeID] = cb_score

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # evaluate - comment out later
    hybrid_rmse_val = hybrid_rmse(testset, anime_index, cf_items_seen, cf_model, alpha)
    print(f"Hybrid RMSE: {hybrid_rmse_val:.4f}")

    return ranked[:top_n]
# for  final usage for user
def hybrid_recommend(user_watched, userID, anime_meta, cf_model, content_sim, anime_index, top_n=10, alpha=0.7):
    # blend CF and CB recommendations
    trainset = cf_model.trainset
    cf_items_seen = {}
    for uid in trainset.all_users():  # numeric inner IDs
        raw_uid = trainset.to_raw_uid(uid)  # convert to original userID
        user_ratings = []
        for iid, rating in trainset.ur[uid]:  # ur maps inner user ID -> list of (item_inner_id, rating)
            raw_iid = trainset.to_raw_iid(iid)
            user_ratings.append((raw_iid, rating))
        cf_items_seen[raw_uid] = user_ratings
    watched_ids = set(aid for aid, _ in user_watched)
    cb_scores = {}
    print("computing CB predictions...")
    for animeID in anime_meta['animeID']:
        if animeID in watched_ids:
            continue
        score = cb_predict(animeID, user_watched, content_sim, anime_index)
        if score is not None:
            cb_scores[animeID] = score

    # normalize CB predictions to match CF rating range
    cb_scores = normalize_cb_predictions(cb_scores)

    final_scores = {}
    # would work if userID was mapped correctly
    # user_in_cf = hasattr(cf_model.trainset, 'to_inner_uid') and str(userID) in trainset._raw2inner_id_users
    # cf_weight = 0.7 if user_in_cf and len(user_watched) >= 5 else 0.0
    # cb_weight = 1 - cf_weight
    print("scoring unseen anime")
    for animeID in anime_meta['animeID']:
        if animeID in watched_ids:
            continue
        cb_score = cb_scores.get(animeID)
        if animeID in cf_items_seen: # blend
            cf_score = cf_model.predict(userID, animeID).est # cf model does guess average if userID not in list
            final_scores[animeID] = alpha * cf_score + (1 - alpha) * (cb_score if cb_score is not None else cf_score)
            # final_scores[animeID] = cf_weight * cf_score + cb_weight * (cb_score if cb_score is not None else cf_score)

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:top_n] + ranked[-(top_n):]

def hybrid_recommend_popularity(anime_avg_rating, anime_count, user_watched, userID, anime_meta, cf_model, content_sim, anime_index, top_n=10, alpha=0.7):
    # blend CF and CB recommendations
    trainset = cf_model.trainset
    cf_items_seen = {}
    for uid in trainset.all_users():  # numeric inner IDs
        raw_uid = trainset.to_raw_uid(uid)  # convert to original userID
        user_ratings = []
        for iid, rating in trainset.ur[uid]:  # ur maps inner user ID -> list of (item_inner_id, rating)
            raw_iid = trainset.to_raw_iid(iid)
            user_ratings.append((raw_iid, rating))
        cf_items_seen[raw_uid] = user_ratings
    watched_ids = set(aid for aid, _ in user_watched)
    cb_scores = {}
    print("computing CB predictions...")
    for animeID in anime_meta['animeID']:
        if animeID in watched_ids:
            continue
        score = cb_predict(animeID, user_watched, content_sim, anime_index)
        if score is not None:
            cb_scores[animeID] = score

    # normalize CB predictions to match CF rating range
    cb_scores = normalize_cb_predictions(cb_scores)

    final_scores = {}
    max_count = max(anime_count.values()) if anime_count else 1
    # would work if userID was mapped correctly
    # user_in_cf = hasattr(cf_model.trainset, 'to_inner_uid') and str(userID) in trainset._raw2inner_id_users
    # cf_weight = 0.7 if user_in_cf and len(user_watched) >= 5 else 0.0
    # cb_weight = 1 - cf_weight
    print("scoring unseen anime")
    for animeID in anime_meta['animeID']:
        if animeID in watched_ids:
            continue
        cb_score = cb_scores.get(animeID)
        pop_score = anime_avg_rating.get(animeID, 7.0)  # fallback rating
        popularity_factor = anime_count.get(animeID, 0) / max_count  # 0..1
        cb_score = 0.7 * (cb_score if cb_score is not None else 7.0) + 0.3 * pop_score * popularity_factor
        final_scores[animeID] = cb_score
        # if animeID in cf_items_seen: # blend
        #     cf_score = cf_model.predict(userID, animeID).est # cf model does guess average if userID not in list (which they won't be)
            # final_scores[animeID] = alpha * cf_score + (1 - alpha) * (cb_score if cb_score is not None else cf_score)
            # final_scores[animeID] = cf_weight * cf_score + cb_weight * (cb_score if cb_score is not None else cf_score)

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:top_n] + ranked[-(top_n):]
    
if __name__ == "__main__":
    if LOAD_ONLY and all(os.path.exists(f) for f in [CF_MODEL_FILE, SIM_MATRIX_FILE, A2IDX_FILE, IDX2A_FILE, BEST_ALPHA_FILE]):
        print("Loading recommender from cache...")
        svd_model = joblib.load(CF_MODEL_FILE)
        # tfidf = joblib.load(TFIDF_FILE)
        # content_sim = np.load(SIM_MATRIX_FILE, allow_pickle=True)
        content_sim = load_npz(SIM_MATRIX_FILE)
        with open(A2IDX_FILE, "rb") as f:
            anime_index = pickle.load(f)
        # with open(IDX2A_FILE, "rb") as f:
        #     idx_to_anime_id = pickle.load(f)
        with open(BEST_ALPHA_FILE, "r") as f:
            best_alpha = float(f.read().strip())
        ratings, anime_meta, trainset, valset, testset = load()
        # cb_scores = precompute_cb_scores(testset, anime_index, content_sim)
    else:
        ratings, anime_meta, trainset, valset, testset = load()

        svd_model = trainSVD(trainset=trainset, testset=testset)

        content_sim, anime_index, idx_to_animeID = trainContentBased(anime_meta)
        # cb_scores = precompute_cb_scores(testset, anime_index, content_sim)
        best_alpha, best_val_rmse = find_best_alpha(valset, svd_model, trainset, content_sim, anime_index)
        print(f"Best alpha: {best_alpha}, Val RMSE: {best_val_rmse:.4f}")
        
        # save
        joblib.dump(svd_model, CF_MODEL_FILE)
        #
        # beacuse sparse
        save_npz(SIM_MATRIX_FILE, content_sim)
        print("saved sim_matrix file")
        with open(A2IDX_FILE, "wb") as f:
            pickle.dump(anime_index, f)
        with open(IDX2A_FILE, "wb") as f:
            pickle.dump(idx_to_animeID, f)
        # exit()
        with open(BEST_ALPHA_FILE, "w") as f:
            f.write(str(best_alpha))

    userID = random.choice(ratings['userID'].unique())
    recommendations = hybrid_recommend_rmse(userID, ratings, trainset, anime_meta, svd_model, content_sim, anime_index, top_n=10, alpha=best_alpha)

    # Map to anime titles
    print(f"Top 10 recommendations for user {userID}:")
    for aid, score in recommendations:
        title = anime_meta.loc[anime_meta['animeID'] == aid, 'title'].values[0]
        print(f"{title} (Score: {score:.4f})")

