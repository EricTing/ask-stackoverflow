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

df = pd.DataFrame({"title": [''],
                    "tags": [''],
                    "paragraphs": ['']})
blank_proba = clf.predict_proba(df)[0, 1]

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/predict')
def predict():
    title = request.args.get('title')
    question = request.args.get('question')
    tags = request.args.get('tags')

    if title is None:
        title = ''
    if question is None:
        question = ''
    if tags is None:
        tags = ''

    df = pd.DataFrame({"title": [title],
                       "tags": [tags],
                       "paragraphs": [question]})
    proba = clf.predict_proba(df)[0, 1] - blank_proba
    if proba < 0:
        proba = 0
    time = 10**reg.predict(df)[0]

    current_tags = tags.split()
    next_tag = rules.get(frozenset(current_tags))

    will_recommend = False
    if next_tag is not None:
        new_df = pd.DataFrame({"title": [title],
                               "tags":
                               [' '.join((tags, ' '.join(next_tag[0])))],
                               "paragraphs": [question]})
        new_proba = clf.predict_proba(new_df)[0, 1] - blank_proba
        if new_proba > proba:
            will_recommend = True

    if will_recommend:
        response = {"proba": proba, "time": time, "next_tag": ' '.join(next_tag[0])}
    else:
        response = {"proba": proba, "time": time, "next_tag": None}
    return jsonify(response)
