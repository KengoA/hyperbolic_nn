import gensim
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy import dot
import os

path = "glove_ep1_size2_lr0.01_vocab200000_poincare_OPTradagrad_COOCCFUNClog_DISTFUNCcosh-dist-sq_bias"
model = gensim.models.Word2Vec.load(path)
wv = model.wv

print(wv)

for i, w in enumerate(wv.index2entity[:10]):
    print("{} {} {} {}".format(i+1, w, wv.vocab[w].count, norm(wv[w])))
    
x = range(len(wv.index2entity))
freq = [wv.vocab[w].count for w in wv.index2entity]

restrict_vocab = 100
fig = plt.figure(1)
plt.plot(x[:restrict_vocab], freq[:restrict_vocab], color="red")

top_words = list(filter(lambda x: x > 10000, freq))
print(len(top_words))
print(freq[5000])