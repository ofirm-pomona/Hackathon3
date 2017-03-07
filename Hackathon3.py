from flask import Flask
from flask import request

from flask_cors import CORS, cross_origin

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import numpy as np

app = Flask(__name__)
CORS(app)

x = [[10, 0, 1, 8], [10, 0, 2, 9], [10, 1, 1, 13], [5, 0, 1, 15], [5, 1, 5, 16], [5, 1, 6, 18]]
y = [8, 7, 5, 6, 9, 9]
degree = 5

model = make_pipeline(PolynomialFeatures(degree), Ridge())

@app.route('/')
def root():
    return app.send_static_file('hackathon3.html')


@app.route('/insert')
def insert():

    road_id = request.args.get('roadId')
    direction = request.args.get('direction')
    day = request.args.get('day')
    time = request.args.get('time')
    status = request.args.get('status')

    x.append([road_id, direction, day, time])
    y.append(status)

    model.fit(np.asarray(x, float), np.asarray(y, float))

    return 'Model updated successfully!'


@app.route('/predict')
def predict():

    road_id = request.args.get('roadId')
    direction = request.args.get('direction')
    day = request.args.get('day')
    time = request.args.get('time')

    status = model.predict([[road_id, direction, day, time]])

    return str(status[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0')