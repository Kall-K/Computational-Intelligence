from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import string
import math

'''
The following code puts together all of the above steps:
1) Tokenize text samples into word uni+bigrams,
2) Vectorize using tf-idf encoding,
3) Select only the top 1000 features from the vector of tokens by discarding tokens that appear fewer than 2 times 
and using f_classif to calculate feature importance.
'''

NGRAM = (1,1)
TOP_K = 40 # i have changed it 
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 2

def ngram_vectorize(train_texts, val_texts, train_labels):
    
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM,  # Use 1-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)
    # vectorizer = TfidfVectorizer()

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)
    
    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')

    return x_train

def import_data():
    file_path = "iphi2802(2).csv" # i have changrd it

    df = pd.read_csv(file_path, encoding='utf-8', delimiter='\t')

    # df['text'] = df['text'].apply(remove_punctuation)
    return df['text'].tolist(), date_in_range(df)

#def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def date_in_range(df):
    date_range = []
    for row in df.itertuples():
        date_range.append((row.date_min, row.date_max))

    return date_range

def normalize_date(date):
    min_date = min(date)
    max_date = max(date)
    return [(x - min_date) / (max_date - min_date) for x in date]

def main():
    texts, dates = import_data()
    labels = list(map(lambda d: math.ceil((d[0] + d[1])/2), dates))
    
    train_texts = ngram_vectorize(texts, texts, labels)
    # print(texts)
    # train_texts = data_preprocess(texts)
    print(type(train_texts))
    # with open('text.txt', "w") as file:
    #     for item in train_texts.toarray():
    #         file.write("%s\n" % item)

    print(len(train_texts.toarray()))
    print(len(train_texts.toarray()[0]))
    print(train_texts.shape)
    # train_texts = MinMaxScaler().fit_transform(train_texts.toarray())
    # with open('text2.txt', "w") as file:
    #     for item in train_texts:
    #         file.write("%s\n" % item)

    # print(labels)
    # print(normalize_date(labels))

    data = {
        'text': train_texts.toarray().tolist(),
        'avg_date': labels,
        'min_date': [d[0] for d in dates],
        'max_date': [d[1] for d in dates]
    }

    df = pd.DataFrame(data)
    df.to_csv('new_data.csv', encoding='utf-8', sep='\t', index=False)

if __name__ == '__main__':
    main()


