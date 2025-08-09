import numpy as np
from scipy.sparse import coo_matrix, save_npz
import pickle
import os

OUT_DIR = "data/processed"
user_idx = np.memmap(os.path.join(OUT_DIR, "user_idx.dat"), dtype=np.int32, mode='r')
anime_idx = np.memmap(os.path.join(OUT_DIR, "anime_idx.dat"), dtype=np.int32, mode='r')
rating_mm = np.memmap(os.path.join(OUT_DIR, "rating.dat"), dtype=np.float32, mode='r')

# load maps to get sizes
with open(os.path.join(OUT_DIR, "user_map.pkl"), "rb") as f:
    user_map = pickle.load(f)
with open(os.path.join(OUT_DIR, "anime_map.pkl"), "rb") as f:
    anime_map = pickle.load(f)

n_users = len(user_map)
n_items = len(anime_map)

print("Building COO matrix: rows(items) x cols(users)")
# implicit expects matrix: shape (n_items, n_users)
# swap axes accordingly
row = anime_idx   # item index
col = user_idx    # user index
data = rating_mm.astype(np.float32)

coo = coo_matrix((data, (row, col)), shape=(n_items, n_users))
csr = coo.tocsr()

print("Saving CSR to disk...")
save_npz(os.path.join(OUT_DIR, "item_user_matrix.npz"), csr)
print("Saved:", os.path.join(OUT_DIR, "item_user_matrix.npz"))
