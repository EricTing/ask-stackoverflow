from flask import request
from flask import render_template
from app import app

import pandas as pd
from sklearn.externals import joblib

clf = joblib.load(
    "/home/ubuntu/Workspace/WhenStackStopsOverFlow/logistic_regression.2015-11-01.pkl")
reg = joblib.load(
    "/home/ubuntu/Workspace/WhenStackStopsOverFlow/linear_regression.2016-01-01.pkl")


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/predict')
def predict():
    title = request.args.get('title')
    question = request.args.get('question')
    tags = request.args.get('tags')

    df = pd.DataFrame({"title": [title],
                       "tags": [tags],
                       "paragraphs": [question],
                       "hasCodes": [1]})
    proba = clf.predict_proba(df)[0, 1]
    time = reg.predict(df)[0]

    # TODO: faked response
    proba = len(title) + len(tags)
    time = len(question)

    return render_template("predict.html",
                           response={"proba": proba,
                                     "time": time})
