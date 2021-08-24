# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 18:16:15 2021

@author: DELL
"""

from flask import Flask, render_template, request
import pickle
import numpy as np

clf = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('tfidf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Message = request.form['Message']
        data = [Message]
        fact = cv.transform(data).toarray()
        my_prediction = clf.predict(fact)
        
        if(int(my_prediction)==1):
            prediction="WOO! It is a hate speech including racism and offensive language."
        else:
            prediction="It's a ordinary message."
        
        return (render_template('index.html', prediction=prediction))
    
if __name__ == '__main__':
    app.run(debug=True)