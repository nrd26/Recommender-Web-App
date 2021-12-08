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
        return render_template('result.html', result = ['https://www.prokerala.com/movies/assets/img/no-poster-available.jpg '])

    URLlist = list()
    ##
    t0 = threading.Thread(target=get_poster,args=(prediction[0],URLlist))
    t1 = threading.Thread(target=get_poster,args=(prediction[1],URLlist))
    t2 = threading.Thread(target=get_poster,args=(prediction[2],URLlist))
    t3 = threading.Thread(target=get_poster,args=(prediction[3],URLlist))
    t4 = threading.Thread(target=get_poster,args=(prediction[4],URLlist))
    t5 = threading.Thread(target=get_poster,args=(prediction[5],URLlist))
    t6 = threading.Thread(target=get_poster,args=(prediction[6],URLlist))
    t7 = threading.Thread(target=get_poster,args=(prediction[7],URLlist))
    t8 = threading.Thread(target=get_poster,args=(prediction[8],URLlist))
    t9 = threading.Thread(target=get_poster,args=(prediction[9],URLlist))
    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()
    t0.join()
    t1.join()
    t2.join()
    t3.join()
    t4.join() 
    t5.join()
    t6.join()
    t7.join()
    t8.join()
    t9.join()
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

