"""
TODO: experiment with averaging, concatenating, and summing word vectors
TODO: late fusion vs. early fusion
TODO: run stemming or lemma-ization to remove multiple variants of same word
TODO: experiment with fasttext, GPT-2, or BERT embeddings (contextual may work better)

Created on Sun Jul 21 2019
Author: Daniel McKinnon
"""

"""Python system modules"""
import re
import os
import json
import time
from operator import add

"""Multiprocessing models to support parallelism"""
import multiprocessing
from multiprocessing import Pool

"""Modules for math"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

"""NLP modules"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')

def create_units_blacklist(units_file='units.json'):
    """
    Create list of physical units to remove from corpus
    """
    with open(units_file) as json_file:
        data = json.load(json_file)
    units = []
    for unit in data:
        for symbol in unit['symbols']:
            units.append(symbol)
    return set(units)

def index_word_vectors():
    """
    Create a dictionary mapping 100-dim GloVe vectors to their words
    """
    BASE_DIR = ''
    GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
    print('Indexing word vectors.')
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def format_text_regex(row):
    """
    Pulls HTML tags, numbers, and punctuation out of
    text fields
    """
    if int(row.name)%1000 == 0:
        print("formatting text.")
    text = re.sub('<[^>]*>|[0-9]|[.,\""\\\/#!$%\^&\*;:{}=\-_`~()]','',row['item_descriptions'])
    text = text + re.sub('<[^>]*>|[0-9]|[.,\""\\\/#!$%\^&\*;:{}=\-_`~()]|\n|\t|\||~|`|‘|•|\xa0','', row['bio'])
    text =  text + row['item_titles']
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(text)
    black_list_words = stop_words | units
    filtered_tokens = [w for w in word_tokens if not w in black_list_words]
    allowed_words = embedded_words.intersection(set(filtered_tokens))
    filtered_tokens = [w for w in filtered_tokens if w in allowed_words]
    return filtered_tokens

def detokenize(row, text_col='text'):
    """
    Merges tokens into a single list.
    This is needed for a step downstream
    """
    if int(row.name)%1000 == 0:
        print("Detokenizing")
    return ' '.join(row[text_col])

def idf_memoization(table, text_field):
    """
    Calculates inverse document frequency.
    This is pulled out as a separate function because it only needs
    to be computed once.  
    """
    print("Calculating memoized idf")
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(table[text_field])
    return tfidf

def tf_idf_words(row, text_field, tfidf, num_words=100):
    """
    Calculated tf-idf for each row
    """
    if int(row.name)%1000 == 0:
        print("Calculating tf-idf")
    text = row[text_field]
    response = tfidf.transform([text])
    feature_names = tfidf.get_feature_names()
    words = {}
    for col in response.nonzero()[1]:
        words[feature_names[col]] = response[0, col]
    try:
        words = pd.Series(words).sort_values(ascending=False)[0:num_words]
        return list(zip(list(words), list(words.index)))
    except Exception as e:
        print(e)
        return ''    

def word2embedding(word):
    """
    Looks up embedding for a given word
    """
    return list(embeddings_index[word])

def embed(row, word_col):
    """
    Calculated the NLP-based embedding of a given row
    """
    if int(row.name)%1000 == 0:
        print("Calculating store embedding")
    embed = [0] * 100
    norm = sum(i[0] for i in row[word_col])
    for _, item in enumerate(row[word_col]):
        norm_embed = [item[0]/norm*x for x in word2embedding(item[1])]
        embed = map(add, norm_embed, embed)
    return np.array(list(embed))

def create_cosine_sim_array(table, embedding_col):
    """
    Use scipy to create a much better than N^2 approximate similarity matrix
    """
    print("Creating cosine simularity array")
    embed_array = np.stack(table[embedding_col])
    cos_sim = cosine_similarity(embed_array)
    return cos_sim

def similarity(row, table, name_col, num_sims=25):
    """
    Write per-row similarity score to df
    """
    top_sims = pd.Series(cos_sim[row.name]).sort_values(ascending=False)[0:num_sims]
    top_sim_stores = list(df.iloc[top_sims.index]['store_url'])
    top_sim_scores = list(top_sims)
    return list(zip(top_sim_stores, top_sim_scores))

def top_ten_sim_brands(row, embed_sim_column):
    """
    Make the top-10 most similar brands more human readable
    """
    if int(row.name)%1000 == 0:
        print("Calculating top-10 matches")
    top_matches = pd.Series(row[embed_sim_column]).sort_values(ascending=False)[0:10]
    return list(zip(list(top_matches), list(top_matches.index)))

def step_1(data):
    """
    First step that can be parallelized
    TODO: give these better names
    """
    data['text'] = data.apply(format_text_regex, axis=1)
    data['plain_text'] = data.apply(detokenize, axis=1)
    data.drop(['item_titles', 'item_descriptions'], axis=1, inplace=True)
    return data

def step_2(data):
    """
    Second step that can be parallelized
    TODO: give these better names
    """
    data['top_100_words'] = data.apply(tf_idf_words, axis=1,\
                                                   args=('plain_text', tfidf))
    data['top_100_words_no_prob'] = data['top_100_words'].apply(lambda row: [x[1] for x in row])
    data['100_words_embed'] = data.apply(embed, axis=1, args =('top_100_words',))
    data.drop(['text','plain_text', 'top_100_words_no_prob'], axis = 1, inplace = True)
    return data

def step_3(data):
    """
    Third step that can be parallelized
    TODO: give these better names
    """
    data['100_words_embed_similarity'] = data.apply(similarity, axis=1, \
                                                args = (df, 'store_url'))
    return data

def parallelize_dataframe(df, func):
    """
    Lets us easily run our computations in parallel
    """
    num_cores = multiprocessing.cpu_count()
    df_split = np.array_split(df, num_cores*4)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

if __name__ == '__main__':
    embeddings_index = index_word_vectors()
    units = create_units_blacklist()
    stop_words = set(stopwords.words('english'))
    embedded_words = set(embeddings_index.keys())
    df = #read text into Pandas DataFrame here
    df = parallelize_dataframe(df, step_1)
    tfidf = idf_memoization(df, 'plain_text')
    df = parallelize_dataframe(df, step_2)
    cos_sim = create_cosine_sim_array(df, '100_words_embed')
    df = parallelize_dataframe(df, step_3)
