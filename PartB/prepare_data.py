import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from numpy.random import randint


token_num = {}

def bin2dec(binary):
    dec = 0
    for b in range(len(binary)):
        dec += binary[-b-1]*pow(2,b)
    return dec

def dec2bin(decimal):
    bin_num = bin(decimal)[2:]
    bin_num = bin_num.zfill(11)
    return [int(b) for b in bin_num]

def check_range(num):
	num = bin2dec(num)
	if num not in range(0,1677):
		num = random_remapping()
	return dec2bin(num)

def random_remapping():
    num = randint(0,1677)
    if num in range(0,1677):
        return num
    else:
        random_remapping()


def import_data():
    file_path = "PartB/iphi2802.csv"

    df = pd.read_csv(file_path, encoding='utf-8', delimiter='\t')

    filtered_df = df[df['region_main_id'] == 1693]
    filtered_df.to_csv('PartB/region1693.csv', index=False, encoding='utf-8')
    texts = filtered_df['text'].values.tolist()
    texts.append('[...] αλεξανδρε ουδις [...]')

    filtered_out = ngram_vectorize(texts)

    with open('PartB/topk_texts.json', 'w', encoding='utf-8') as fp:
        json.dump(cos_similarity(filtered_out), fp, ensure_ascii=False)

    if os.path.exists('PartB/token_id.json'):
        os.remove('PartB/token_id.json')
    with open('PartB/token_id.json', 'w', encoding='utf-8') as fp:
        json.dump(token_num, fp, ensure_ascii=False)
    

def ngram_vectorize(x):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(x)
    # here all tokens are stored with their ID
    # so I know which number from 0 to 1677 corresponds to each word
    feature_names = vectorizer.get_feature_names_out()
    counter=0
    for i, x in enumerate(x):
        feature_index = X[i,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [X[i, x] for x in feature_index])
        for feature_index, score in tfidf_scores:
            if feature_names[feature_index] not in token_num.values():
                token_num.update({counter:feature_names[feature_index]})
                counter +=1
    return X


def cos_similarity(tfidf_matrix):
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    similarities = cosine_sim_matrix[-1][:-1]
    return [{'id':i,'similarity':cos} for i,cos in enumerate(similarities) if cos!=0]

import_data()

