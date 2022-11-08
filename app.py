import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

__data_columns = None

model = pickle.load(open('model.pkl','rb'))
with open('columns.json', 'r') as f:
    __data_columns = json.load(f)['data_columns']


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]

    try:
        loc_index = __data_columns.index(data[len(data) - 1])
    except:
        print
        loc_index = -1
        print(-1)

    x = [0] * len(__data_columns)

    x[0] = data[0]
    x[1] = data[1]
    x[2] = data[2]
    x[3] = data[3]
    x[4] = data[4]
    if loc_index >= 0:
        x[loc_index] = 1

    x = np.array(x)
    x = x.reshape(1, -1)
    print(x)

    output = model.predict(x)[0]
    output = round(output, 2)
    return render_template("home.html",prediction_text=("The estimated price is {}".format(output)) + " RUB")


if __name__ == "__main__":
    app.run(debug=True)

