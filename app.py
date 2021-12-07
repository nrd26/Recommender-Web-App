import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from prediction import movie_predict
import json
import http.client
import threading

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    title = request.form.to_dict()
    print(title)
    prediction = movie_predict(title['movie_name'])
    # if prediction == 'Movie not in dataset':
    #     URLlist = prediction
    if type(prediction) == type(0):
        print("MOVIES NOT IN THE DATASET")
        return render_template('result.html', result = ['https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/fb3ef66312333.5691dd2253378.jpg'])

    URLlist = list()
    ##
    t0 = threading.Thread(target=get_poster,args=(prediction[0],URLlist))
    t1 = threading.Thread(target=get_poster,args=(prediction[0],URLlist))
    t2 = threading.Thread(target=get_poster,args=(prediction[0],URLlist))
    t3 = threading.Thread(target=get_poster,args=(prediction[0],URLlist))
    t4 = threading.Thread(target=get_poster,args=(prediction[0],URLlist))
    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t0.join()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    ##
    # for i in prediction:
    #     URLlist.append(get_poster(i))
    # print(URLlist)
    return render_template('result.html', result = URLlist)

#returns URL of the poster, "-1" on error
def get_poster(movie_name,URLlist):
    conn = http.client.HTTPSConnection("imdb-internet-movie-database-unofficial.p.rapidapi.com")
    headers = {
    'x-rapidapi-host': "imdb-internet-movie-database-unofficial.p.rapidapi.com",
    'x-rapidapi-key': "0b9c3bb71cmsha990fbcbca0e8e6p1a5894jsnc0101b1bb302"
    }
    movie_name = movie_name.replace(' ','_')
    try:
        conn.request("GET", "/search/"+movie_name, headers=headers)
        res = conn.getresponse()
        data = res.read()
        data = data.decode("utf-8")
        info = json.loads(data)
        URLlist.append(info["titles"][0]["image"])
        return
    except:
        return "-1"

if __name__ == '__main__':
    app.run(debug=True, port=5000)

