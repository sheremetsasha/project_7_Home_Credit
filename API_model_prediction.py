#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:50:46 2023

@author: alexandrasheremet
"""

import streamlit as st
from flask import Flask,  jsonify

import pandas as pd
import matplotlib.pyplot as plt
from shap import TreeExplainer
#import plotly.express as px
import json
import seaborn as sns
import pickle 


#Create an interface Flask class that is WSGI application


app = Flask(__name__)

html_header = """
    <head>
        <title>Interactive Dashboard - Agreement for bank loan</title>
        <meta charset = "utf-8">
        <meta name = "keywords" content = "Bank Loan, Dashboard, Scoring">
        <meta name = "description" content="Application - Dashboard - Credit Scoring">
        <meta name = "author" content = "Alexandra Sheremet">
        <meta name = "viewport" content = "width=device-width, initial-scale=1">
    </head>             
    <h1 style = "font-size:300%; color: #007ea8; font-family:Arial"> Bank loan decision <br>
        <h2 style = "color: #007ea8; font-family: Georgia"> </h2>
        <hr style = "  display: block;
          margin-top: 0;
          margin-bottom: 0;
          margin-left: auto;
          margin-right: auto;
          border-style: inset;
          border-width: 1.5px;"/>
     </h1>
"""

st.markdown(html_header, unsafe_allow_html = True)

#-------------------------------------------
# load the model from disk
#-------------------------------------------
print("Loading model...")
with open('finalized_model.sav', 'rb') as f:
    best_model = pickle.load(f)
    
#-------------------------------------------
# load the data for prediction
#-------------------------------------------
print("Loading data...")
with open('data_clients.pickle', 'rb') as f:
    
    data = pickle.load(f)
    
@app.route("/loading_data", methods = ["GET"])
def reading_data():
    
    print("Loading data")
    data_to_file = data.copy()
    obj = {"data": data_to_file.values.tolist(),
           "features": list(data_to_file.columns),
           "index": list(data_to_file.index)}
    
    return json.dumps(obj)
    
#-------------------------------------------
# prediction of the model
#-------------------------------------------
@app.route("/start_model", methods = ["GET"])
def model_prediction():
    
    print("Model prediction...")
    y_pred_lgbm = best_model.predict(data)
    
    return jsonify(list(y_pred_lgbm))

#-------------------------------------------
#local features with SHAP
#-------------------------------------------
@app.route("/shap", methods = ["GET"])
def shap():
    print("Local features analyse...")
    explainer_cust = TreeExplainer(best_model)
    shap_values_cust = explainer_cust(data)

    obj = {
        "expexted_value" : explainer_cust.expected_value[0].tolist(),
        "values" :         shap_values_cust.values[:,:,1].tolist(),
        "base_values":     shap_values_cust.values[:,:,1].tolist()
        }
    
    return json.dumps(obj)



#-------------------------------------------
#Global feature importance
#-------------------------------------------
@app.route("/glob_features", methods = ["GET"])
def glob_feat():
    feat_import_glob = pd.DataFrame()
    feat_import_glob_sup = data.copy()
    feat_import_glob['feature'] = feat_import_glob_sup.columns
    import_glob = best_model.feature_importances_
    tot_glob = import_glob.sum()
    feat_import_glob['importance'] = import_glob/tot_glob
    feat_import_glob.sort_values(by = "importance", ascending = False)
    
    return feat_import_glob.to_json(orient = 'values')


feat_import_glob = pd.read_json(glob_feat(), orient ='values')
feat_import_glob.columns =['feature', 'importance']


################################################
#Display in the website
################################################

# Information on the dataset
st.subheader('Statistical Information about the Data')

st.write('For the analysing creditworthiness of customers, the API uses LGBM model with the score 70%')

st.write('Test dataset has ', len(data.columns), 'features and ',  len(data), ' elements')

st.subheader('Global feature importance characterizing the model')
fig_glob = plt.figure(figsize=(4,3))
sns.set(font_scale = 1.5)
sns.barplot(data = feat_import_glob.sort_values(by = "importance", ascending = False)[:5], x = "importance", y = "feature")
#plt.title("Global features importance")

def get_feature_plot():    
    sns.set(font_scale = 1.5)
    sns.barplot(data = feat_import_glob.sort_values(by = "importance", ascending = False)[:5], x = "importance", y = "feature")
    #plt.title("Global features importance")

st.pyplot(fig_glob)



if __name__ == "__main__":
    app.run(port = 5000)