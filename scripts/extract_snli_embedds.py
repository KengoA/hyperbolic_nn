import pickle
import numpy as np
import matplotlib.pyplot as plt

words = pickle.load(open('id_to_word', 'rb'))
words_set = set(words)

poincare_words = {}
in_em = []
out_em = []
with open('skipgram.txt', 'r') as f:
    f.readline()
    l = f.readline()

    while l:
        l = l.strip().split(' ')

        if l[0] in words_set:
            poincare_words[l[0]] = list(map(lambda x: float(x), l[1 :]))
        
        l = f.readline()

embedds = []
count_w = 0
count_t = 0
for w in words:
    count_t += 1
    if w in poincare_words:
        count_w += 1
        embedds.append(poincare_words[w])
        in_em.append(poincare_words[w])
    else:
        x = np.random.uniform(-0.001, 0.001, 2)
        embedds.append(np.random.uniform(-0.001, 0.001, 2))
        out_em.append(x)

print(count_w / count_t)
print(count_w)
print(count_t)
in_em = np.array(in_em)
out_em = np.array(out_em)
plt.scatter(in_em[:, 0], in_em[:, 1], c = 'r')
plt.scatter(out_em[:, 0], out_em[:, 1], c = 'b')
plt.show()
pickle.dump(embedds, open('2d_snli_embedds_skipgram', 'wb'))
