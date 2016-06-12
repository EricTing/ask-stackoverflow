from flask import Flask
from flask.ext.bootstrap import Bootstrap
app = Flask(__name__)
bootstrap = Bootstrap(app)
from app import views

# from sklearn.externals import joblib

# clf = joblib.load(
#     "/home/ubuntu/Workspace/WhenStackStopsOverFlow/logistic_regression.pkl")
# reg = joblib.load(
#     "/home/ubuntu/Workspace/WhenStackStopsOverFlow/linear_regression.pkl")
