from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math
import numpy as np
'''
The following code puts together all of the above steps:
1) Tokenize text samples into word uni+bigrams,
2) Vectorize using tf-idf encoding,
3) Select only the top 1000 features from the vector of tokens by discarding tokens that appear fewer than 7 times 
and using f_classif to calculate feature importance.
'''

NGRAM = (1,1)
TOP_K = 1000 
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 7

def ngram_vectorize(train_texts, train_labels):
    
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

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)
    
    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')

    return x_train

def import_data():
    file_path = "iphi2802.csv" 

    df = pd.read_csv(file_path, encoding='utf-8', delimiter='\t')

    return df['text'].tolist(), df['region_main_id'].tolist(), date_in_range(df)

def date_in_range(df):
    # contanation of min & max date to a tuple
    # append all the tuple to a list with date ranges
    date_range = []
    for row in df.itertuples():
        date_range.append((row.date_min, row.date_max))

    return date_range

def normalize_date_range(date_ranges):
    #find the min date & the max date of the pairs with dates
    min_date = min(min(pair) for pair in date_ranges)
    max_date = max(max(pair) for pair in date_ranges)

    return [((start - min_date) / (max_date - min_date), 
             (end - min_date) / (max_date - min_date))
            for start, end in date_ranges]

def main():
    # extraction of texts and date ranges from the dataset
    texts, region_id, dates = import_data()

    # finding the avg of each pair of dates to use it as label 
    labels = list(map(lambda d: math.ceil((d[0] + d[1])/2), dates))
    
    # call function to tokenize and vectorize texts
    train_texts = ngram_vectorize(texts, labels)

    min_value = train_texts.min()
    max_value = train_texts.max()

    print("Minimum value:", min_value)
    print("Maximum value:", max_value)

    print(type(train_texts))
    print(train_texts.shape)

    # 1st dataframe with normalized range of dates and the avg of each pair of date range
    df_T1 = pd.DataFrame({
        'region_id': region_id,
        'min_date': [d[0] for d in dates],
        'max_date': [d[1] for d in dates]
    })

    # 2nd dataframe with vectorized texts
    T2 = train_texts.toarray()
    df_T2 = pd.DataFrame(T2)

    # concatanation of the 2 frames
    df = pd.concat([df_T2, df_T1], axis=1)

    # Remove rows where the first 1000 values are equal to zero
    zero_rows_indices = df.iloc[:, :TOP_K].eq(0).all(axis=1)
    df = df[~zero_rows_indices]

    # export to csv file 
    df.to_csv('preprocessed_data.csv', encoding='utf-8', sep='\t', index=False)


if __name__ == '__main__':
    main()


