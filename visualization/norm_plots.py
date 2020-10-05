# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 10:56:59 2017

@author: suny2
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:46:52 2017

@author: suny2
"""
import scipy.io as sio
import numpy as np
import pandas as pd
import json
import glob
import os
from pprint import pprint
import re

from scipy.sparse.extract import find

#%%
wordId_df = pd.read_csv('data/wordIDHash.csv', header=0, names=['id', 'word', 'unknown'])
word2Id = pd.Series(wordId_df.id.values,index=wordId_df.word).to_dict()

times = range(0,26) # total number of time points (20/range(27) for ngram/nyt)
# emb_all = sio.loadmat('results/emb_frobreg10_diffreg50_symmreg10_iter10.mat')
# emb_all = sio.loadmat('embeddings/embeddings_0.mat')

def load_embeddings_map(embeddings_dir):
    to_load = glob.glob(os.path.join(embeddings_dir, "*"))
    mat_objects = [ sio.loadmat(p)['U'] for p in to_load]
    embeddings_year = [ int(re.findall(r'\d+', os.path.basename(p))[0]) for p in to_load]
    return { k:m for k,m in  zip(embeddings_year, mat_objects)}

emb = sio.loadmat('embeddings/embeddings_10.mat')

print emb

emb_all = load_embeddings_map('embeddings')
#%%

# print emb_all

words = ['cell','telephone', 'apple', 'banana']
# words = ['thou','chaise','darwin','telephone']
allnorms = []
for w in words:
    norms = []
    for year in times:
        # emb = emb_all['U_%d' % times.index(year)][word2Id[w],:]
        emb = emb_all[year][word2Id[w]]
        norms.append(np.linalg.norm(emb))
    
    norms = np.array(norms)
    norms = norms / sum(norms)
    allnorms.append(norms)

#%%
corr = np.corrcoef(allnorms, rowvar=True)
print corr
#%%
import matplotlib.pyplot as plt
import pickle
#Z = sio.loadmat('tsne_output/%s_tsne.mat'%word)['emb']
#list_of_words = pickle.load(open('tsne_output/%s_tsne_wordlist.pkl'%word,'rb'))
years = [t*10 for t in times]
markers = ['+','o','x','*']
plt.clf()
for k in xrange(len(allnorms)):
    norms = allnorms[k]
    plt.plot(years,norms,marker=markers[k],markersize=7)
plt.legend(words)
plt.xlabel('year')
plt.ylabel('word norm')
plt.show()

#sio.savemat('tsne_output/%s_tsne.mat'%word,{'emb':Z})
#pickle.dump(list_of_words,open('tsne_output/%s_tsne_wordlist.pkl'%word,'wb'))