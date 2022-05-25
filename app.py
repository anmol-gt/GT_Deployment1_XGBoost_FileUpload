from crypt import methods
from flask import Flask, render_template, redirect, url_for, request
import pickle
import numpy as np
import pandas as pd
# from requests import request
import xgboost as XGBRegressor
# import sys
# import logging


# create flask app name and error logging
app = Flask(__name__)
# app.logging.addHandler(logging.StreamHandler(sys.stdout))
# app.logger.setLevel(logging.ERROR)




# create prediction function
def predict_results(data):
    # load the pickle file
    modelfile = open('XGB_pickle.pkl', 'rb')
    res = pickle.load(modelfile)
    input = pd.read_csv(data)
    preds = res.predict(input)
    preds_df1 = pd.DataFrame({"Hours Avalibility":preds})
    preds_df1['Hours Availibility Rounded'] = preds_df1['Hours Avalibility'].apply(np.ceil)
    result_df = pd.concat([input,preds_df1], axis=1)
    return result_df



# create homepage routing
@app.route('/')
def home():
    return render_template('home.html')


# create prediction routing
@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['data_file']
    if data.filename != '':
        df = predict_results(data)
        return render_template('results.html', tables=[df.to_html()], titles=['na', 'Prediction Hours'])

    return "Error Occurred."

if __name__ == '__main__':
    app.run(debug=True)

