import pandas as pd
import math
import difflib 
import random

#stop_words.txt contains stop words
fhand = open('stop_words.txt','r')
stopwords_list = (fhand.read()).split()
fhand.close()
df = pd.read_excel('dataset/dataset.xlsx') #read from excel
strength = 20 #quality of predictions, directly proportional to processing time
inpt_name = None

#If a title absent from db is entered, this function returns the closest match
def find_closest(inpt_title):
    title_list = df['title'].tolist()
    for i in range(0,len(title_list)):
        title_list[i] = str(title_list[i])
    title = (difflib.get_close_matches(str(inpt_title),list(title_list)))[0]
    return title

#Removes stopwords from the passed text
def remove_stopwords(text_list):
    list_append = list()
    for i in text_list:
        if i not in stopwords_list:
            list_append.append(i.strip(' '))
    rand_list = list()
    for i in range(0,strength):
        rand_index = random.randrange(len(list_append)-100,len(list_append)-20)
        rand_list.append(list_append[rand_index])
    return list_append[:strength] + rand_list 

#Returns the movie-id for the movie_name
def get_movie_id(movie_name):
    try:
        movie_id = int((df['movie_id'][df['title']==movie_name]))
    except:
        movie_id = -1
    return movie_id

#Returns all the non-stop words
def good_words(movie_id):
    string = None
    for i in (df['meta_data'][df['movie_id']==movie_id]):
        string = i
    if string is None:
        return inpt_name.split(' ')*3
    good_words = remove_stopwords(string.split())
    good_words = set(good_words)
    good_words = list(good_words)
    return good_words + inpt_name.split(' ')*3

def IDF_helper(x,good_word,idf_dict):
    if good_word in str(x):
        idf_dict[good_word] = idf_dict.get(good_word,0)+1

#Calculates the inverse document frequency
def IDF(idf_dict,good_words_list):
    for word in good_words_list:
        df['meta_data'].apply(IDF_helper,args=[word,idf_dict])
    for i,j in idf_dict.items():
        idf_dict[i]=math.log(df.shape[0]/j,10)


#Calculates term frequency and multiplies it with idf
def TF_helper(x,good_word,tf_dict,idf_dict):
    xlist = x.split()
    tf_dict[xlist[0]] = tf_dict.get(xlist[0],0) + (x.count(good_word)/len(xlist))*(idf_dict.get(good_word,0))

#Calculates the Term frequency 
def TF(tf_dict,good_words_list,idf_dict):
    for word in good_words_list:
        df['meta_data'].apply(TF_helper,args=[word,tf_dict,idf_dict])

#Returns a list of similar movies, returns -1 if unable to find similar movies
def movie_predict(movie_name):
    global inpt_name
    movie_name = movie_name.lower()
    inpt_name = movie_name
    idf_dict = dict()
    tf_dict = dict()
    tf_idf_list = list()
    movie_id = get_movie_id(movie_name)
    if movie_id == -1:
        try:
            movie_name = find_closest(movie_name)
            movie_id = get_movie_id(movie_name)
        except:
            movie_name = inpt_name
    print(movie_name)
    good_words_list = good_words(movie_id)  #good words are words other than stop words   
    print(good_words_list)  
    IDF(idf_dict,good_words_list)
    TF(tf_dict,good_words_list,idf_dict)
    tup_list = list()
    for i,j in tf_dict.items(): #j is movie_id and i is the tf*idf
        tup_list.append((j,i))
    tup_list.sort(reverse=True) #Sort in decreasing order of tf*idf
    movie_list = list()
    for i,j in tup_list[:20]:   #Get top 10 results
        for i in (df['title'][df['movie_id']==float(j)]):
            s_temp = str(i).strip(' ')
            if s_temp != '' and s_temp != 'nan' and s_temp != ' ' and s_temp!=movie_name:
                movie_list.append(i)
    return movie_list[:10] #Return 10 similar movies

    

