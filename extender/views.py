from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import os
from gensim.models import Word2Vec
import gensim
import numpy as np

from random import sample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def download_gensim_model():
    model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top100.zh.model') 
    # model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top50.zh.model') 
    # model_path = os.path.join(settings.MEDIA_ROOT, 'patent_non_stops_top20.zh.model') 
    model = Word2Vec.load(model_path)
    return model

def index(request):
    return render(request, "extender/index.html", context=None)

@csrf_exempt
def search(request, term="測試"):
    if request.method == "GET":
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
    
