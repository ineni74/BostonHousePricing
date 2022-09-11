import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('boston_regressor.pkl', "rb"))
scaler = pickle.load(open('scaling_pickle.pkl', "rb"))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    print("predict starting...")
    data = request.json['data']
    print("given data...")
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    input_data = scaler.transform(np.array(data).reshape(1,-1))
    print(input_data)
    output = model.predict(input_data)[0]
    print(output)
    return render_template('home.html', prediction_text="The House price Prediction is : {}".format(output))
    


if __name__=="__main__":
    app.run(debug=True)
    
    
