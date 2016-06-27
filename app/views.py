from flask import request
from flask import jsonify
from flask import render_template
from app import app

import pandas as pd
import pickle
from sklearn.externals import joblib

clf = joblib.load(
    "/home/ubuntu/Workspace/WhenStackStopsOverFlow/combined_model.2016-02-01.pkl")
reg = joblib.load(
    "/home/ubuntu/Workspace/WhenStackStopsOverFlow/combined_model_time.2016-02-01.pkl")
rules = pickle.load(open(
    "/home/ubuntu/Workspace/WhenStackStopsOverFlow/tags.2016-02-01.profile.pkl"))


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


def norm(proba):
    if proba < 0.47:
        return 0
    else:
        return proba - 0.47


@app.route('/predict')
def predict():
    title = request.args.get('title')
    paragraphs = request.args.get('body')
    tags = request.args.get('tags')

    print(title)
    print(paragraphs)
    print(tags)

    response = {"proba": 0, "suggestions": None}
    if any([title == '', paragraphs == '', tags == '']):
        print("no prediction")
        return jsonify(response)
    else:
        print("prediction")
        # current probability
        df = pd.DataFrame({"title": [title],
                           "tags": [tags],
                           "paragraphs": [paragraphs]})

        proba = clf.predict_proba(df)[0, 1]
        print("proba:", proba)
        proba = norm(proba)
        response['proba'] = proba

        # suggestions
        current_tags = frozenset(tags.split())
        next_tags = [s for s in rules
                     if current_tags.issubset(s) and s != current_tags]

        if len(next_tags) > 0:
            next_df = pd.DataFrame({"tags": [' '.join(s) for s in next_tags]})
            next_df['title'] = title
            next_df['paragraphs'] = paragraphs
            next_proba = clf.predict_proba(next_df)[:, 1]
            next_df['next_proba'] = next_proba
            next_df['next_proba'] = next_df['next_proba'].apply(norm)
            next_df = next_df[next_df['next_proba'] > proba]
            next_df = next_df.sort_values('next_proba', ascending=False)
            suggestions = dict(zip(next_df.head()['tags'], next_df.head()[
                'next_proba']))
            response["suggestions"] = suggestions
        else:
            pass

        print(response)

        return jsonify(response)
