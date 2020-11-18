# emojifier
This is a sentiment analysis project incorporating NLP, Deep Learning, RNN, Flask, AWS.
1. Extracted online labeled text dataset as training set, converted text into word vector representation with the help of pre-trained 50-dimensional Glove word embeddings.
2. Train the dataset using Bi-Directional LSTM model in Keras.
3. Built the web app using Flask and packaged the app with Docker.
4. Deployed the [web app](http://ec2-3-15-65-183.us-east-2.compute.amazonaws.com:5000/) in AWS EC2. 