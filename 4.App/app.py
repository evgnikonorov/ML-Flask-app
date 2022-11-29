import joblib
import numpy as np

from flask import Flask, render_template, request, url_for, flash, redirect
app = Flask(__name__)


messages = [{'title': 'IW',
             'content': 'Please input IW parameter',
		 'default': '45'},
            {'title': 'IF',
             'content': 'Please input IF parameter',
		 'default': '140'},
            {'title': 'VW',
             'content': 'Please input VW parameter',
		 'default': '8'},
            {'title': 'FP',
             'content': 'Please input FP parameter',
		 'default': '80'}
            ]

@app.route('/')
def index():
    return render_template('index.html', messages=messages)

@app.route('/predict2', methods=('POST', 'GET'))
def predict2():
	if request.method == 'POST':
		title = request.form['title']
		content = request.form['content']

		if not title:
			flash('Title is required!')
		elif not content:
			flash('Content is required!')
		else:
			messages.append({'title': title, 'content': content})
	X = np.array([ 45. , 140. ,   8. ,  80. ]).reshape(1, -1)
	model =joblib.load('rf_model_jl.jl') 
	pred = model.predict(X)
	return str(pred[0,1])
	

@app.route('/predict', methods = ['POST'])
def hello_world():
	IW = float(request.form['IW'])
	IF = float(request.form['IF'])
	VW = float(request.form['VW'])
	FP = float(request.form['FP'])
		
	X = np.array([ IW , IF ,   VW ,  FP ]).reshape(1, -1)
	model =joblib.load('rf_model_jl.jl') 
	pred = model.predict(X)
	W = str(pred[0,1])
	D = str(pred[0,0])
	return str('Width: '+ W + ';  Depth: ' + D)