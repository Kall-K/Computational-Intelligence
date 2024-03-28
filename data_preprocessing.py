from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd
import numpy as np
import string
import math

'''
The following code puts together all of the above steps:
1) Tokenize text samples into word uni+bigrams,
2) Vectorize using tf-idf encoding,
3) Select only the top 20,000 features from the vector of tokens by discarding tokens that appear fewer than 2 times 
and using f_classif to calculate feature importance.
'''

# Vectorization parameters

# suggests to use unigram, so i have to change the following
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM = (1,1)

# Limit on the number of features. We use the top 1K features.
TOP_K = 1000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

def ngram_vectorize(train_texts, train_labels):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
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

    print(x_train.toarray())

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    return x_train

#===================================================
def data_preprocess(documents):
    # Bag-of-Words (BoW) using CountVectorizer
    count_vectorizer = CountVectorizer()
    bow_matrix = count_vectorizer.fit_transform(documents)

    print("Bag-of-Words (BoW) representation:")
    print(bow_matrix.toarray())
    print("Vocabulary:")
    print(count_vectorizer.get_feature_names_out())

    # TF-IDF using TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    print("\nTF-IDF representation:")
    print(tfidf_matrix.toarray())
    print("Vocabulary:")
    print(tfidf_vectorizer.get_feature_names_out())
#===================================================

def import_data():
    file_path = "iphi2802.csv"

    df = pd.read_csv(file_path, encoding='utf-8', delimiter='\t')

    df['text'] = df['text'].apply(remove_punctuation)

    return df['text'].tolist(), date_in_range(df)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def date_in_range(df):
    date_range = []
    for row in df.itertuples():
        date_range.append((row.date_min, row.date_max))

    return date_range

if __name__ == '__main__':
    texts, dates = import_data()
    labels = list(map(lambda d: math.ceil((d[0] + d[1])/2), dates))
    train_texts = ngram_vectorize(texts, labels)
    # print(train_texts.toarray(),type(train_texts))
    # print(texts)
    # data_preprocess(texts)

    with open('text.txt', "w") as file:
        # Iterate over the list and write each element to the file
        for item in train_texts:
            file.write("%s\n" % item)
