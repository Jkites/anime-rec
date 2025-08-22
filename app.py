from flask import Flask, request, render_template
from recommender.model import get_recommendations
import gzip
import pandas as pd
import xml.etree.ElementTree as ET
# import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'data/uploaded'

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "user_file" not in request.files:
            print("Nothing found")
            return "No file uploaded", 400
        file = request.files["user_file"]
        if file.filename == "":
            return "No file selected", 400

        if file.filename.endswith(".gz"):
            with gzip.open(file.stream, "rb") as f:
                xml_bytes = f.read()
            xml_data = xml_bytes.decode("utf-8")
        else:
            xml_data = file.read().decode("utf-8")
        
        # filename = secure_filename(file.filename)
        # filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        # file.save(filepath)

        user_id, username, user_watched = parse_mal_xml(xml_data)
        # os.remove(filepath)

        recs = get_recommendations(user_watched, user_id)
        # temp
        # recs = [{
        #     "title": "title1",
        #     "score": 2,
        #     "image_url": "https://cdn.myanimelist.net/s/common/uploaded_files/1455542152-1164a6a65b3efde6f0d5be12cf67edfc.png",
        #     "url": "https://myanimelist.net/anime/1328",
        #     "global_score": 5
        # },{"title": "title2"},{"title": "title3"},{"title": "title4"},{"title": "title5"},{"title": "title6"},{"title": "title7"},{"title": "title8"},
        # {"title": "title9"},
        # {"title": "title10"},
        # {"title": "title11"},
        # {"title": "title12"},
        # {"title": "title13"},
        # {"title": "title14"},
        # {"title": "title15"},
        # {"title": "title16"},
        # {"title": "title17"},
        # {"title": "title18"},
        # {"title": "title19"},
        # {"title": "title20"}]

        return render_template("index2.html", recommendations=recs, user = username)

    return render_template("upload.html")

def parse_mal_xml(xml_path):
    # tree = ET.fromstring(xml_path)
    root = ET.fromstring(xml_path)

    user_info = root.find("myinfo")
    user_id = int(user_info.find("user_id").text)
    username = user_info.find("user_name").text

    user_watched = []
    for anime in root.findall("anime"):
        anime_id = int(anime.find("series_animedb_id").text)
        score = anime.find("my_score").text
        score = int(score) if score and score.isdigit() else None

        if score is not None and score > 0:  # skip entries with no score/0 score
            user_watched.append((anime_id, score))
            # print(str(user_id) + str(anime_id) + " " + str(score))

    return user_id, username, user_watched
if __name__ == '__main__':
    app.run(debug=True)