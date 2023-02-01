#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 21:40:51 2023

@author: alexandrasheremet
"""

from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle
import json


app_model = Flask(__name__)


@app_model.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    prediction = np.array2string(model.predict(data))

    return jsonify(prediction)

if __name__ == '__main__':
    modelfile = 'finalized_model.sav'
    model = pickle.load(open(modelfile, 'rb'))
    app_model.run(debug=True, host='0.0.0.0')