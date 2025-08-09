import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import csv
import pickle

CSV_PATH = "data/ratings.csv"
OUT_DIR = "data/processed"        # OUT
CHUNKSIZE = 1_000_000             # higher is faster

os.makedirs(OUT_DIR, exist_ok=True)

# build mapping of unique user/anime to index
user_map = {}
anime_map = {}
next_user = 0
next_anime = 0
total_rows = 0

print("Building maps")
with pd.read_csv(CSV_PATH, chunksize=CHUNKSIZE, usecols=['userID','animeID','rating']) as it:
    for chunk in tqdm(it):
        total_rows += len(chunk)
        # factorize the chunk for speed
        users = chunk['userID'].unique()
        for u in users:
            if u not in user_map:
                user_map[u] = next_user
                next_user += 1
        animes = chunk['animeID'].unique()
        for a in animes:
            if a not in anime_map:
                anime_map[a] = next_anime
                next_anime += 1

print(f"Total rows: {total_rows:,}, unique users: {len(user_map):,}, unique anime: {len(anime_map):,}")

# save maps to disk for reverse lookup later
with open(os.path.join(OUT_DIR, "user_map.pkl"), "wb") as f:
    pickle.dump(user_map, f)
with open(os.path.join(OUT_DIR, "anime_map.pkl"), "wb") as f:
    pickle.dump(anime_map, f)

# allocate memmaps
print("Allocating memmaps...")
user_idx_path = os.path.join(OUT_DIR, "user_idx.dat")
anime_idx_path = os.path.join(OUT_DIR, "anime_idx.dat")
rating_path = os.path.join(OUT_DIR, "rating.dat")

user_idx = np.memmap(user_idx_path, dtype=np.int32, mode='w+', shape=(total_rows,))
anime_idx = np.memmap(anime_idx_path, dtype=np.int32, mode='w+', shape=(total_rows,))
rating_mm = np.memmap(rating_path, dtype=np.float32, mode='w+', shape=(total_rows,))

# fill memmaps
print("Filling memmaps...")
pos = 0
with pd.read_csv(CSV_PATH, chunksize=CHUNKSIZE, usecols=['userID','animeID','rating']) as it:
    for chunk in tqdm(it):
        n = len(chunk)
        
        chunk['user_idx'] = chunk['userID'].map(user_map).astype(np.int32)
        chunk['anime_idx'] = chunk['animeID'].map(anime_map).astype(np.int32)
        chunk['rating'] = chunk['rating'].astype(np.float32)

        user_idx[pos: pos+n] = chunk['user_idx'].values
        anime_idx[pos: pos+n] = chunk['anime_idx'].values
        rating_mm[pos: pos+n] = chunk['rating'].values

        pos += n

# flush memmaps
user_idx.flush()
anime_idx.flush()
rating_mm.flush()

print("Files written to:", OUT_DIR)
