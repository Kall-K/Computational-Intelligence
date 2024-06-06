import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from numpy.random import randint#######
import csv

NGRAM = (1,1) 
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 1 # all the words

token_idf = {}
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

def test():
    n_pop = 5
    n_bits = 22
    pop = []
    for _ in range(n_pop):
        new = randint(0, 2, n_bits).tolist()
        pop.append(check_range(new[:11])+check_range(new[11:]))

    print(pop)
    for p in pop:
        print(bin2dec(p[:11]),bin2dec(p[11:]))
    similarity(pop[0])
    # bin = randint(0, 2, 22).tolist() 
    # first_bin = bin[:11]
    # second_bin = bin[11:]
    # print(bin)

    # first_num = bin2dec(first_bin)
    # print(first_num)

    # second_num = bin2dec(second_bin)
    # print(second_num)
    # check_range(second_num)

    # print("after range check:")
    # first_num = check_range(first_num)
    # second_num = check_range(second_num)
    # print(first_num)
    # print(second_num)
    # print('decimal to bin:')
    # print(dec2bin(first_num))
    # print(dec2bin(second_num))

def import_data():
    file_path = "PartB/iphi2802.csv"

    df = pd.read_csv(file_path, encoding='utf-8', delimiter='\t')

    filtered_df = df[df['region_main_id'] == 1693]
    texts = filtered_df['text'].values.tolist()
    texts.append('[...] αλεξανδρε ουδις [...]')

    filtered_out = ngram_vectorize(texts)

    print(cos_similarity(filtered_out))
    with open('PartB/topk_texts.json', 'w', encoding='utf-8') as fp:
        json.dump(cos_similarity(filtered_out), fp, ensure_ascii=False)

    pd.DataFrame(filtered_out.toarray()).to_csv("PartB/tfidf_matrix.csv", index=False)

    print(len(token_idf),len(token_num))

    os.remove('PartB/token_tfidf.json')
    with open('PartB/token_tfidf.json', 'w', encoding='utf-8') as fp:
        json.dump(token_idf, fp, ensure_ascii=False)
    
    os.remove('PartB/token_id.json')
    with open('PartB/token_id.json', 'w', encoding='utf-8') as fp:
        json.dump(token_num, fp, ensure_ascii=False)


def ngram_vectorize(x):
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM,  # Use 1-grams.
            'strip_accents': 'unicode',
            'decode_error': 'ignore',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY
    }

    vectorizer = TfidfVectorizer(**kwargs)
    X = vectorizer.fit_transform(x)

    counter=0

    feature_names = vectorizer.get_feature_names_out()
    for i, x in enumerate(x):
        # print(f"Document {i+1}:")
        feature_index = X[i,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [X[i, x] for x in feature_index])
        for feature_index, score in tfidf_scores:
            # print(f"{feature_names[feature_index]}: {score}")
            if feature_names[feature_index] not in token_num.values():
                token_num.update({counter:feature_names[feature_index]})
                counter +=1
            token_idf.update({feature_names[feature_index]: score})
    return X


def cos_similarity(tfidf_matrix):
    specific_text_vector = tfidf_matrix[-1]
    cosine_similarities = cosine_similarity(tfidf_matrix[:-1], specific_text_vector)
    cosine_similarities = cosine_similarities.flatten()

    return [{'id':i+1,'similarity':cos} for i,cos in enumerate(cosine_similarities) if cos!=0]

def load_top_k():
	with open('PartB/topk_texts.json', 'r', encoding='utf-8') as f:
		top_k = json.load(f)

	rows = [0]
	for rec in top_k:
		rows.append(rec['id'])

	# df = pd.read_csv('PartB/region1693.csv', skiprows = lambda x: x not in rows)
	df = pd.read_csv('PartB/region1693.csv', skiprows = lambda x: x not in rows)
	return df['text'].values.tolist()
	# texts.append('[...] αλεξανδρε ουδις [...]')
	
	# return df_vect

def load_token(dec_num):
    with open('PartB/token_id.json', 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    # print(tokens[str(dec_num)])
    # if dec_num in tokens:
    return tokens.get(str(dec_num))
    # else:
    #     return None

def ngram_vectorizer(x):
	# Create keyword arguments to pass to the 'tf-idf' vectorizer.
	kwargs = {
			'ngram_range': NGRAM,  # Use 1-grams.
			'strip_accents': 'unicode',
			'decode_error': 'ignore',
			'analyzer': TOKEN_MODE,  # Split text into word tokens.
			'min_df': MIN_DOCUMENT_FREQUENCY
	}

	vectorizer = TfidfVectorizer(**kwargs)
	X = vectorizer.fit_transform(x)
	return X

def similarity(x):
	# score = 0
    texts = load_top_k()
    first = load_token(bin2dec(x[:11]))
    second = load_token(bin2dec(x[11:]))
    text = first + ' αλεξανδρε ουδις ' + second
    
    texts.append(text)
    print(texts)
    tfidf_matrix = ngram_vectorizer(texts)
    # print(tfidf_matrix)
    specific_text_vector = tfidf_matrix[-1]
    cosine_similarities = cosine_similarity(tfidf_matrix[:-1], specific_text_vector)
    cosine_similarities = cosine_similarities.flatten()
    print(cosine_similarities)
    print(sum(cosine_similarities)/len(cosine_similarities))
    return sum(cosine_similarities)/len(cosine_similarities)
	# return [{'id':i+1,'similarity':cos} for i,cos in enumerate(cosine_similarities) if cos!=0]
	# return [{'id':i+1,'similarity':cos} for i,cos in enumerate(cosine_similarities)]





test()
import_data()

