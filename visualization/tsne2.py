# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09  2017


"""
import scipy.io as sio
import numpy as np
import json
from pprint import pprint
from scipy.spatial.distance import pdist
import pickle
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def read_words_reverse_index():
    wordId_df = pd.read_csv('data/wordIDHash.csv', header=0, names=['id', 'word', 'unknown'])
    wordlist = wordId_df.word.values
    word2Id_all = pd.Series(wordId_df.id.values,index=wordId_df.word).to_dict()
    word2Id = word2Id_all
    return word2Id_all

def read_embeddings():
    # loading prepared_embeddings object
    with open('results/prepared_embeddings.pkl', 'r') as f:
        emb_all = pickle.load(f)
    return emb_all

def reverse_dict(my_map):
    inv_map = {v: k for k, v in my_map.iteritems()}
    return inv_map

def get_closest_words_list(word, emb, year, word2Id, id2word, list_size=10):
    embnrm = np.reshape(np.sqrt(np.sum(emb**2,1)),(emb.shape[0],1))
    emb_normalized = np.divide(emb, np.tile(embnrm, (1,emb.shape[1])))           
    v = emb_normalized[word2Id[word],:]

    dist_metric =np.dot(emb_normalized,v)
    idx = np.argsort(dist_metric)[::-1]    
    return [id2word[i] for i in idx[:list_size]], dist_metric[idx]


def project_embeddings_to_plane(embeddings, dim=2):
    embeddings = np.array(embeddings)
    num_slices, list_len, _ = embeddings.shape
    flattened_embeddings = np.array(embeddings).reshape(-1,50)
    model = TSNE(n_components=dim, metric = 'euclidean')
    tsne_projections =  model.fit_transform(flattened_embeddings)
    return tsne_projections.reshape(num_slices, list_len, dim)

def get_words_labels(words):
    labels = []
    for t, t_words in enumerate(words):
        labels.extend([ "(%s, %d)" % (word, t) for word in t_words])
    return labels

def print_trjectories(tsne_projections, words, target_word, dims=2):

    flattened_projections = tsne_projections.reshape(-1,dims)
    flattened_words = words.reshape(-1)
    words_labels = get_words_labels(words)

    target_word_idx = np.argwhere(flattened_words == target_word)
    target_word_projections = flattened_projections[target_word_idx].reshape(-1,dims)
    print(target_word_projections.shape)

    plt.figure()
    plt.plot(flattened_projections[:,0], flattened_projections[:,1], 'b.')
    for x,y, label in zip(flattened_projections[:,0], flattened_projections[:,1], words_labels):
        plt.text(x, y, label)
    plt.plot(target_word_projections[:,0], target_word_projections[:,1], 'r-')
    plt.title("Trajetoria e vizinhanca da palavra '%s' no corpus" % target_word)    
    plt.show()

target_word = "amazon"
times = range(0,26)
project_dims = 2
word_idx = read_words_reverse_index()
idx_word = reverse_dict(word_idx)
embeddings = read_embeddings()
closest_words_list_len = 10

close_embeddings = []
close_words = []
for t in times:

    time_slice_embeddings = embeddings['U_%d' % t]

    close_words_on_slice, words_dist = get_closest_words_list(
        target_word, time_slice_embeddings, t, word_idx, idx_word, closest_words_list_len)

    close_words_embeddings_on_slice = [ time_slice_embeddings[word_idx[w]] for w in close_words_on_slice]
    
    close_embeddings.append(close_words_embeddings_on_slice)
    close_words.append(close_words_on_slice)

close_words = np.array(close_words).reshape(-1,closest_words_list_len)
tsne_projection = project_embeddings_to_plane(close_embeddings, project_dims)

print_trjectories(tsne_projection, close_words, target_word, project_dims)






