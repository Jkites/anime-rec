import pandas as pd


df = pd.read_csv("data/animes.csv")
df2 = pd.read_csv("data/ratings.csv")

df["mal_id"] = df["mal_url"].str.split("/").str[-1].astype(int)
id_map = dict(zip(df["animeID"], df["mal_id"]))
df2["animeID"] = df2["animeID"].map(id_map)
df2 = df2.dropna(subset=["animeID"])
df["mal_id"] = df["mal_url"].str.split("/").str[-1].astype(int)
df["animeID"] = df["mal_id"]
df = df.drop(columns=["mal_id"])
df.to_csv("data/updated_animes.csv", index=False)
df2.to_csv("data/updated_ratings.csv", index=False)
print("successfully converted")