from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import os
from gensim.models import Word2Vec
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from random import sample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from pymongo import MongoClient
conn = MongoClient(settings.MONGO_URL)
db = conn.patent


import jieba
jieba.set_dictionary(os.path.join(settings.MEDIA_ROOT, 'dict.txt.big'))
# with open(os.path.join(settings.MEDIA_ROOT, 'stopwords.txt'),'r', encoding='utf-8') as f:
#     stopwords = f.read().split('\n')


def download_gensim_model():
    # model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top50.zh.model') 
    try:
        model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top100.zh.model') 
        # model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top50.zh.model') 
        # model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top20.zh.model') 
        model = Word2Vec.load(model_path)
    except:
        model_path = os.path.join(settings.MEDIA_ROOT, 'patent_non_stops_top20.zh.model') 
        model = Word2Vec.load(model_path)
    return model


def retrieve_titles(terms, next):
    model = download_gensim_model()
    title_vector = np.zeros((400,))
    for t in terms:
        if t in model.wv.vocab:
            title_vector += model.wv[t]

    if np.array_equal(title_vector, np.zeros((400,))):
        relevant_titles = []
    else:
        title_vectors = np.load(os.path.join(settings.MEDIA_ROOT, 'downloaded', 'title_vectors.npy'))
        scores = cosine_similarity(np.array(title_vector).reshape(1, -1), title_vectors)[0]
        retrieved = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[next:next+50]
        relevant_titles = []
        for i in retrieved:
            relevant_titles.append(db.patent.find({'_id':i})[0])
    return relevant_titles


def index(request):
    return render(request, "extender/index.html", context=None)

@csrf_exempt
def search_relevant_terms(request, term="測試"):
    if request.method == "GET":  # deal with only one word
        model = download_gensim_model()
        return_terms = dict(model.most_similar(positive=term.strip()))
        return JsonResponse(return_terms) 
    if request.method == "POST":
        data = request.POST
        lines = data['terms_str'].split('\n')
        related_terms = []

        model = download_gensim_model()
        for line in lines:
            terms = line.split(',') 
            terms_in_vocab = [t for t in terms if t in model.wv.vocab]            
            if len(terms_in_vocab) != 0:
                return_terms = list(map(lambda x:x[0].strip(), model.most_similar(positive=terms_in_vocab)))
                related_terms.extend(return_terms)

        if len(related_terms) == 0:
            related_terms.append('很抱歉，您檢索的關鍵字並未在過去的專利文件中出現過!')

        return JsonResponse({"related_terms":list(set(related_terms))}) 

@csrf_exempt
def search_relevant_titles(request, title="", next=0):
    if request.method == "GET": # deal with only title
        terms = jieba.cut(title, cut_all=True)
        relevant_titles = retrieve_titles(terms, next)

    if request.method == "POST":
        data = request.POST
        terms = np.hstack(np.array([line.split(',') for line in data['terms_str'].split('\n')]))
        print(terms)
        relevant_titles = retrieve_titles(terms, next)

    response_dict = {
        "data":relevant_titles
    }

    return JsonResponse(response_dict) 




def visualize(request):
    model = download_gensim_model()
    vocabs = sample(list(model.wv.vocab), 300)
    X = model[vocabs]

    n_clusters = 4
    cls_array = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    
    # max_len_index = 0
    # max_len = 0
    # for i in range(n_clusters):
    #     if sum(cls_array==i) > max_len:
    #         max_len_index = i  
    #         max_len = sum(cls_array==i)

    # X_reduced = PCA(n_components=2).fit_transform(X[cls_array==max_len_index])
    # vocabs = np.array(vocabs)[cls_array==max_len_index]

    # X_reduced = PCA(n_components=2).fit_transform(X)
    # vocabs = np.array(vocabs)

    # return render(request, "extender/visualize.html", context={"vectors":X_reduced.tolist(), 'vocabs':vocabs.tolist(), 'classes':cls_array})
    return render(request, "extender/visualize.html", context={"vectors":X.tolist(), 'vocabs':vocabs, 'classes':cls_array.tolist()})

def evaluator(request):
    return render(request, "extender/evaluator.html", context={})
    
