from flask import request
from flask import render_template
from app import app

import pandas as pd
from sklearn.externals import joblib

clf = joblib.load(
    "/home/ubuntu/Workspace/WhenStackStopsOverFlow/nb_tags.2016-02-01.product.pkl")
reg = joblib.load(
    "/home/ubuntu/Workspace/WhenStackStopsOverFlow/combined_model_time.2016-02-01.pkl")


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
    time = 10 ** reg.predict(df)[0]

    return render_template("predict.html",
                           response={"proba": str(proba),
                                     "time": str(time)})
