import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise.dump import dump, load
from surprise.accuracy import rmse

# Directories
CACHE_DIR = "data/processed"
os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_SAVE_DIR = "recommender"
# FILES
RAW_CSV = "data/ratings.csv"
MODEL_FILE = os.path.join(MODEL_SAVE_DIR, "svd_model.pkl")
USER_MAP_FILE = os.path.join(CACHE_DIR, "user_map.pkl")
ANIME_MAP_FILE = os.path.join(CACHE_DIR, "anime_map.pkl")
# SVD Parameters !!!
N_FACTORS=50
N_EPOCHS=10
LR_ALL=0.005
REG_ALL=0.02

# load CSV, encode user/anime IDs to integer indices, and save mapping files.
def load_and_preprocess():
    if os.path.exists(USER_MAP_FILE) and os.path.exists(ANIME_MAP_FILE):
        print("Loading saved ID maps...")
        # with open(USER_MAP_FILE, "rb") as f:
        #     user_map = pickle.load(f)
        # with open(ANIME_MAP_FILE, "rb") as f:
        #     anime_map = pickle.load(f)
        df = pd.read_csv(RAW_CSV)
    else:
        print("Maps not found, create them first using create_maps.py")
        # df = pd.read_csv(RAW_CSV)

        # userIDs, user_map = pd.factorize(df["userID"])
        # animeIDs, anime_map = pd.factorize(df["animeID"])

        # df["userID"] = userIDs
        # df["animeID"] = animeIDs

        # with open(USER_MAP_FILE, "wb") as f:
        #     pickle.dump(user_map, f)
        # with open(ANIME_MAP_FILE, "wb") as f:
        #     pickle.dump(anime_map, f)
    print("loaded")
    return df
# train an SVD model
def train_model(df, n_factors=N_FACTORS, n_epochs=N_EPOCHS, lr_all=LR_ALL, reg_all=REG_ALL):
    # reader = Reader(rating_scale=(0.1, 10.0))
    # print("Data set loading from df")
    # data = Dataset.load_from_df(df[["userID", "animeID", "rating"]], reader)
    # print("Splitting data...")
    # trainset, testset = train_test_split(data, test_size=0.2, random_state=1)
    # print("Training model...")

    # Shuffle
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    print("Shuffled data")
    # Test split
    test_size_abs = int(0.2 * len(df))
    test_df = df.iloc[:test_size_abs]
    train_val_df = df.iloc[test_size_abs:]

    # Validation split from train_val
    val_size_abs = int(0.1 * len(train_val_df))
    val_df = train_val_df.iloc[:val_size_abs]
    train_df = train_val_df.iloc[val_size_abs:]

    print(f"Train: {len(train_df):,} rows")
    print(f"Val: {len(val_df):,} rows")
    print(f"Test: {len(test_df):,} rows")

    reader = Reader(rating_scale=(0.1, 10.0))
    print("Dataset loading from df...")
    trainset = Dataset.load_from_df(train_df[['userID', 'animeID', 'rating']], reader).build_full_trainset() # out of memory
    valset = list(val_df[['userID', 'animeID', 'rating']].itertuples(index=False, name=None))
    testset = list(test_df[['userID', 'animeID', 'rating']].itertuples(index=False, name=None))

    print("training model")
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, verbose=True)
    algo.fit(trainset)
    print("Validation set:")
    val_preds = algo.test(valset)
    val_rmse = rmse(val_preds, verbose=True)
    print("Evaluating on test set...")
    predictions = algo.test(testset)
    test_rmse = rmse(predictions, verbose=True)

    # save model
    dump(MODEL_FILE, algo=algo)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    return algo

# def cross_validate_model(df, n_factors=100, n_epochs=20):
#     """
#     Runs 5-fold cross-validation.
#     """
#     reader = Reader(rating_scale=(0.1, 10.0))
#     data = Dataset.load_from_df(df[["userID", "animeID", "rating"]], reader)
#     algo = SVD(n_factors=n_factors, n_epochs=n_epochs)
#     results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#     return results

# loads model from file
def load_model():
    file_path, algo = load(MODEL_FILE)
    return algo

# predicts rating
def predict_rating(user_raw, anime_raw):
    with open(USER_MAP_FILE, "rb") as f:
        user_map = pickle.load(f)
    with open(ANIME_MAP_FILE, "rb") as f:
        anime_map = pickle.load(f)

    algo = load_model()

    try:
        user_inner = list(user_map).index(user_raw)
        anime_inner = list(anime_map).index(anime_raw)
    except ValueError:
        return None  # User or anime not in training data

    prediction = algo.predict(user_inner, anime_inner)
    return prediction.est

if __name__ == "__main__":
    df = load_and_preprocess()

    # train and save model
    algo, rmse = train_model(df)

    # results = cross_validate_model(df)
