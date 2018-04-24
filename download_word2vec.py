import os 
import boto3
from gensim.models import Word2Vec
import json

MEDIA_ROOT = 'media'
# with open('pwd.json') as f:
#     pwd_data = json.load(f)
pwd_data = {
    "AWS_ACCESS_KEY_ID":os.environ['AWS_ACCESS_KEY_ID'],
    "AWS_SECRET_ACCESS_KEY":os.environ['AWS_SECRET_ACCESS_KEY'],
    "S3_BUCKET":os.environ['S3_BUCKET']
}

AWS_ACCESS_KEY_ID = pwd_data['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = pwd_data['AWS_SECRET_ACCESS_KEY']
S3_BUCKET = pwd_data['S3_BUCKET']

if 'downloaded' not in os.listdir(MEDIA_ROOT):
    os.mkdir(os.path.join(MEDIA_ROOT, 'downloaded'))

# file_names = ["patent_non_stops_top100.zh.model", "patent_non_stops_top100.zh.model.syn1neg.npy", "patent_non_stops_top100.zh.model.wv.syn0.npy", "title_vectors.npy"]
file_names = ["patent_non_stops_top191.zh.model", "patent_non_stops_top191.zh.model.syn1neg.npy", "patent_non_stops_top191.zh.model.wv.syn0.npy", "title_vectors_top191.npy"]
# file_names = ["patent_non_stops_top50.zh.model", "patent_non_stops_top50.zh.model.syn1neg.npy", "patent_non_stops_top50.zh.model.wv.syn0.npy"]
# file_names = ["patent_non_stops_top20.zh.model", "patent_non_stops_top20.zh.model.syn1neg.npy", "patent_non_stops_top20.zh.model.wv.syn0.npy"]
try:
    for n in file_names:
        if n not in os.listdir(os.path.join(MEDIA_ROOT, 'downloaded')): 
            s3 = boto3.resource('s3')
            s3.meta.client.download_file(S3_BUCKET, "patent/" + n, os.path.join(MEDIA_ROOT, 'downloaded', n))
    print("download success!")
except:
    print('download fail!')


# from subprocess import call
# call(["gunicorn", "patent_query_extedter.wsgi"])

# web: gunicorn patent_query_extender.wsgi
