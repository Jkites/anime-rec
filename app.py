from flask import Flask, request, render_template
import pandas as pd
# !!!
# from recommender.model import recommend_for_user

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        new_user_df = pd.read_csv(file)
        # recommendations = recommend_for_user(new_user_df)
        recommendations = None
        return render_template('index.html', recs=recommendations)
    return render_template('index.html', recs=None)
'''
import gzip
import shutil

with gzip.open('features.xml.gz') as f:

    features = pd.read_csv(f)
https://docs.python.org/3/library/xml.etree.elementtree.html
features.head()
'''
if __name__ == '__main__':
    app.run(debug=True)