from flask import Flask
app = Flask(__name__)
from app import views

from sklearn.externals import joblib

clf = joblib.load(
    "/home/ubuntu/Workspace/WhenStackStopsOverFlow/logistic_regression.pkl")
reg = joblib.load(
    "/home/ubuntu/Workspace/WhenStackStopsOverFlow/linear_regression.pkl")
