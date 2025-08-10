import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
#import argparse
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz, load_npz
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
#import math
from sklearn.metrics import root_mean_squared_error

RAW_CSV = "data/ratings.csv" # filepath
CHUNK_SIZE = 1_000_000
N_FOLDS = 5
CACHE_DIR = "data/processed" # folder
MODEL_SAVE_DIR = "recommender" # folder
TEST_RATIO = 0.1
# ALS params !!! TUNE LATER
FACTORS = 50 # 
ITERATIONS = 15 # 

# assign rating to a fold.
def assign_split(user_id, anime_id, k_folds, final_test_ratio):
    final_test_hash = hash((user_id, anime_id, "final")) % 100 # 0 to 99
    if final_test_hash < final_test_ratio * 100:
        return "final_test", None # assign test
    else:
        fold_id = hash((user_id, anime_id)) % k_folds # 0 to k_folds - 1
        return "cv", fold_id # assign fold from training

# divide the data
def preprocess(csv_path, k_folds, final_test_ratio, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    # check if already exists
    if all(os.path.exists(os.path.join(cache_dir, f"user_map.pkl")) and
           os.path.exists(os.path.join(cache_dir, f"anime_map.pkl")) and
           os.path.exists(os.path.join(cache_dir, f"fold_{i}_train.npz")) and
           os.path.exists(os.path.join(cache_dir, f"fold_{i}_test.npz")) and
           os.path.exists(os.path.join(cache_dir, f"final_test.npz"))
           for i in range(k_folds)):
        print("Processed data already exists.")
        return

    print("Preprocessing raw CSV in chunks...")

    user_map = {}
    anime_map = {}
    next_user_idx = 0
    next_anime_idx = 0

    # create fold buffers
    fold_train_rows, fold_train_cols, fold_train_data = [[] for _ in range(k_folds)], [[] for _ in range(k_folds)], [[] for _ in range(k_folds)]
    fold_test_rows, fold_test_cols, fold_test_data = [[] for _ in range(k_folds)], [[] for _ in range(k_folds)], [[] for _ in range(k_folds)]
    final_test_rows, final_test_cols, final_test_data = [], [], []

    for chunk in tqdm(pd.read_csv(csv_path, chunksize=CHUNK_SIZE)):
        for _, row in chunk.iterrows():
            user_id = row["userID"]
            anime_id = row["animeID"]
            rating = row["rating"]

            if user_id not in user_map:
                user_map[user_id] = next_user_idx
                next_user_idx += 1
            if anime_id not in anime_map:
                anime_map[anime_id] = next_anime_idx
                next_anime_idx += 1

            u_idx = user_map[user_id]
            a_idx = anime_map[anime_id]

            split_type, fold_id = assign_split(user_id, anime_id, k_folds, final_test_ratio)

            if split_type == "final_test":
                final_test_rows.append(a_idx)
                final_test_cols.append(u_idx)
                final_test_data.append(rating)
            else:
                if hash((u_idx, a_idx, "test")) % 5 == 0:   # 1/5 of fold is validation
                    fold_test_rows[fold_id].append(a_idx)
                    fold_test_cols[fold_id].append(u_idx)
                    fold_test_data[fold_id].append(rating)
                else:                                       # else training
                    fold_train_rows[fold_id].append(a_idx)
                    fold_train_cols[fold_id].append(u_idx)
                    fold_train_data[fold_id].append(rating)

    # save mappings
    pickle.dump(user_map, open(os.path.join(cache_dir, "user_map.pkl"), "wb"))
    pickle.dump(anime_map, open(os.path.join(cache_dir, "anime_map.pkl"), "wb"))

    shape = (len(anime_map), len(user_map))

    # save folds
    for i in range(k_folds):
        save_npz(os.path.join(cache_dir, f"fold_{i}_train.npz"),
                 coo_matrix((fold_train_data[i], (fold_train_rows[i], fold_train_cols[i])), shape=shape))
        save_npz(os.path.join(cache_dir, f"fold_{i}_test.npz"),
                 coo_matrix((fold_test_data[i], (fold_test_rows[i], fold_test_cols[i])), shape=shape))

    # save final test set
    save_npz(os.path.join(cache_dir, "final_test.npz"),
             coo_matrix((final_test_data, (final_test_rows, final_test_cols)), shape=shape))

    print("Preprocessing complete.")

# evaluate model
# def evaluate(model, test_matrix):
#     preds = model.user_factors @ model.item_factors.T
#     # Clip predictions to rating range
#     preds = np.clip(preds, 1, 10)
#     true_rows, true_cols = test_matrix.nonzero()
#     errors = []
#     for r, c in zip(true_rows, true_cols):
#         errors.append((preds[c, r] - test_matrix[r, c]) ** 2)
#     return math.sqrt(np.mean(errors))
def evaluate(model, test_matrix):
    users, items = test_matrix.nonzero()
    preds = []
    trues = []
    for u, i in zip(users, items):
        preds.append(model.user_factors[u] @ model.item_factors[i])
        trues.append(test_matrix[u, i])
    return root_mean_squared_error(trues, preds)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--csv", required=True)
    # parser.add_argument("--cache_dir", default="cache")
    # parser.add_argument("--factors", type=int, default=50)
    # parser.add_argument("--iterations", type=int, default=15)
    # parser.add_argument("--folds", type=int, default=5)
    # parser.add_argument("--final_test_ratio", type=float, default=0.1)
    # args = parser.parse_args()

    preprocess(RAW_CSV, N_FOLDS, TEST_RATIO, CACHE_DIR)

    # load mappings
    user_map = pickle.load(open(os.path.join(CACHE_DIR, "user_map.pkl"), "rb"))
    anime_map = pickle.load(open(os.path.join(CACHE_DIR, "anime_map.pkl"), "rb"))

    # cross-validation
    rmses = []
    for i in range(N_FOLDS):
        train = load_npz(os.path.join(CACHE_DIR, f"fold_{i}_train.npz")).tocsr()
        test = load_npz(os.path.join(CACHE_DIR, f"fold_{i}_test.npz")).tocsr()

        model = AlternatingLeastSquares(factors=FACTORS, iterations=ITERATIONS)
        model.fit(train)
        rmse = evaluate(model, test)
        print(f"Fold {i} RMSE: {rmse:.4f}")
        rmses.append(rmse)

    print(f"Mean CV RMSE: {np.mean(rmses):.4f}")

    # final test evaluation
    # train on all CV folds combined
    train_all = sum((load_npz(os.path.join(CACHE_DIR, f"fold_{i}_train.npz")).tocsr() for i in range(N_FOLDS)))
    final_test = load_npz(os.path.join(CACHE_DIR, "final_test.npz")).tocsr()
    final_model = AlternatingLeastSquares(factors=FACTORS, iterations=ITERATIONS)
    final_model.fit(train_all)
    final_rmse = evaluate(final_model, final_test)
    print(f"Final test RMSE: {final_rmse:.4f}")

    # Save final model
    pickle.dump(final_model, open(os.path.join(MODEL_SAVE_DIR, "als_model.pkl"), "wb"))
    print("Final model saved.")


if __name__ == "__main__":
    main()