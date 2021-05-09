import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from prediction import movie_predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    title = request.form.to_dict()
    print(title)
    prediction = movie_predict(title['movie_name'])

    return render_template('home.html', prediction_text="Movie recommendations {}".format(prediction[0:10]))


if __name__ == '__main__':
    app.run(debug=True, port=5000)