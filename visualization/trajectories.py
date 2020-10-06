# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09  2017


"""
from prepare_embeddings import main
import scipy.io as sio
import numpy as np
import json
from pprint import pprint
from scipy.spatial.distance import pdist
import pickle
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

PROJECTION_DIMS = 2

def read_words_reverse_index(reverse_index_file_path):
    # wordId_df = pd.read_csv('data/wordIDHash.csv', header=0, names=['id', 'word', 'unknown'])
    wordId_df = pd.read_csv(reverse_index_file_path, header=0, names=['id', 'word', 'unknown'])
    wordlist = wordId_df.word.values
    word2Id_all = pd.Series(wordId_df.id.values,index=wordId_df.word).to_dict()
    word2Id = word2Id_all
    return word2Id_all

def read_embeddings(embeddings_file_path):
    # loading prepared_embeddings object
    # with open('results/prepared_embeddings.pkl', 'r') as f:
    with open(embeddings_file_path, 'r') as f:
        emb_all = pickle.load(f)
    return emb_all

def reverse_dict(my_map):
    inv_map = {v: k for k, v in my_map.iteritems()}
    return inv_map

def get_closest_words_list(word, emb, word2Id, id2word, list_size=10):
    embnrm = np.reshape(np.sqrt(np.sum(emb**2,1)),(emb.shape[0],1))
    emb_normalized = np.divide(emb, np.tile(embnrm, (1,emb.shape[1])))           
    v = emb_normalized[word2Id[word],:]

    dist_metric =np.dot(emb_normalized,v)
    idx = np.argsort(dist_metric)[::-1]    
    return [id2word[i] for i in idx[:list_size]], dist_metric[idx]


def project_embeddings_to_plane(embeddings):
    embeddings = np.array(embeddings)
    num_slices, list_len, _ = embeddings.shape
    flattened_embeddings = np.array(embeddings).reshape(-1,50)
    model = TSNE(n_components=PROJECTION_DIMS, metric = 'euclidean')
    tsne_projections =  model.fit_transform(flattened_embeddings)
    return tsne_projections.reshape(num_slices, list_len, PROJECTION_DIMS)

def get_words_labels(words):
    labels = []
    for t, t_words in enumerate(words):
        labels.extend([ "(%s, %d)" % (word, t) for word in t_words])
    return labels

def print_trjectories(tsne_projections, words, target_word):

    neighborhood_size = words.shape[1] - 1

    flattened_projections = tsne_projections.reshape(-1,PROJECTION_DIMS)
    flattened_words = words.reshape(-1)
    words_labels = get_words_labels(words)

    target_word_idx = np.argwhere(flattened_words == target_word)
    target_word_projections = flattened_projections[target_word_idx].reshape(-1,PROJECTION_DIMS)

    fig = plt.figure()
    plt.plot(flattened_projections[:,0], flattened_projections[:,1], 'b.')
    for x,y, label in zip(flattened_projections[:,0], flattened_projections[:,1], words_labels):
        plt.text(x, y, label)
    plt.plot(target_word_projections[:,0], target_word_projections[:,1], 'r-')
    plt.title("Trajetoria e vizinhanca da palavra '%s' no corpus" % target_word)    
    fig.savefig('results/%s_trajectory_ns%d.png' % (target_word, neighborhood_size))
    plt.show()

def get_target_word_closest_neighborhood_for_each_time_slice(target_word, times, embeddings, word_idx, list_len):
    idx_word = reverse_dict(word_idx)
    close_embeddings = []
    close_words = []
    for t in times:
        time_slice_embeddings = embeddings['U_%d' % t]
        close_words_on_slice, words_dist = get_closest_words_list(
            target_word, time_slice_embeddings, word_idx, idx_word, list_len)
        close_words_embeddings_on_slice = [ time_slice_embeddings[word_idx[w]] for w in close_words_on_slice]
        close_embeddings.append(close_words_embeddings_on_slice)
        close_words.append(close_words_on_slice)
    
    close_words = np.array(close_words).reshape(-1,list_len)
    return close_embeddings, close_words

def main(embeddings_file_path, reverse_index_path, target_word, closest_words_list_len):

    word_idx = read_words_reverse_index(reverse_index_path)
    idx_word = reverse_dict(word_idx)
    embeddings = read_embeddings(embeddings_file_path)
    times = range(0,len(embeddings.keys()))

    close_embeddings, close_words = get_target_word_closest_neighborhood_for_each_time_slice(
        target_word, times, embeddings, word_idx, closest_words_list_len)
    
    tsne_projection = project_embeddings_to_plane(close_embeddings)

    print_trjectories(tsne_projection, close_words, target_word)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embeddings-file', dest='embeddings_file_path', type=str,
                        default="results/prepared_embeddings.pkl")
    parser.add_argument('--reverse-index-file', dest='reverse_index_file', type=str,
                        default="data/wordIDHash.csv")
    parser.add_argument('--target-word', dest='target_word', type=str)
    parser.add_argument('--closest-words-list-len', dest='closest_words_list_len', type=int,
                        default=10)

    args = parser.parse_args()
    main(args.embeddings_file_path, args.reverse_index_file, args.target_word, args.closest_words_list_len)