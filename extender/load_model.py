from gensim.models import Word2Vec
from django.conf import settings
import os
import numpy as np

# word2vec_model
def download_gensim_model():
    # model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top50.zh.model') 
    try:
        model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top191.zh.model') 
        # model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top100.zh.model') 
        # model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top50.zh.model') 
        # model_path = os.path.join(settings.MEDIA_ROOT, 'downloaded', 'patent_non_stops_top20.zh.model') 
        model = Word2Vec.load(model_path)
    except:
        model_path = os.path.join(settings.MEDIA_ROOT, 'patent_non_stops_top20.zh.model') 
        model = Word2Vec.load(model_path)
    print("model:", model_path)
    return model

word2vec_model = download_gensim_model()


# title_vectors
title_vectors = np.load(os.path.join(settings.MEDIA_ROOT, 'downloaded', 'title_vectors_top191.npy'))

