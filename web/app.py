import sys
from flask import Flask,render_template,request,jsonify
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np


app = Flask(__name__)

def get_model():
	global model,tokenizer
	model = keras.models.load_model('emoji_model.h5')
	tokenizer = pickle.load(open('tokenizer.pickle','rb'))
	print('Model Loaded!!')



@app.route('/')
def home():
	return render_template('view.html')

@app.route('/predict',methods = ['POST'])
def predict():
	maxlen = 50
	text = request.form['input_text']
	print('The received text is: %s' %(text))
	test_sent = tokenizer.texts_to_sequences([text])
	test_sent = keras.preprocessing.sequence.pad_sequences(test_sent, maxlen = maxlen)
	pred = model.predict(test_sent)
	response = {'prediction': int(np.argmax(pred))}
	return jsonify(response)


if __name__ == "__main__":
	get_model()
	app.run(host="0.0.0.0", port=5000,debug=False)