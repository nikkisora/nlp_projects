import pandas as pd
import numpy as np
import re


def get_seq_prob(dic, seq, V, alpha, ngr):
    """aff"""
    if seq[:-1] not in dic.index:
        return 1/V
    c = dic.loc[seq[:-1]].groupby(str(ngr)).sum()
    c_w = c.get(seq[-1], 0)
    if not c_w:
        return 1/V
    return (c_w + alpha) / (c.sum() + alpha*V)


def get_perp(dic, sen, V, alpha, ngr):
    """asdf"""
    if len(sen) <= ngr:
        return get_seq_prob(dic, sen, V, alpha, len(sen))
    p_w = 1
    for i in range(len(sen) - ngr + 1):
        p_w *= get_seq_prob(dic, sen[i: i+ngr], V, alpha, ngr)
    return (pow(1/p_w, 1/len(sen)))


# Create dataset
files = ['l2/texts/neuroman.txt', 'l2/texts/song.txt']
corp = ''

for f in files:
    with open(f) as file:
        corp += file.read()

corp = re.sub(r'[^A-Za-z!?;\'. ]', ' ', corp) # remove unneeded characters
corp = re.sub(r' +', ' ', corp) # remove multi-spaces
corp = re.sub(r'[!?;.]+', ' . \n _', corp) # Change all sentence endings to single period and add _ as sentence begining

parags = corp.split('\n')

df = pd.Series(parags)

train_size = 0.999

train = df.sample(frac=train_size, random_state=1)
test = df.drop(train.index)

# training
train_text = ' '.join(train)
train_word_list = train_text.split()

n_grams = []
n = 10

for k in range(len(train_word_list)-n+1):
    n_grams.append(tuple(train_word_list[k:k+n]))

df = pd.DataFrame(n_grams, columns=[str(i+1) for i in range(n)])

get_n_grams = lambda df, n: df.groupby(df.columns.tolist()[:n]).\
                              size().\
                              reset_index().\
                              rename(columns={0:'num'})

dic = get_n_grams(df, n).set_index([str(i+1) for i in range(n-1)]).sort_index()

# testing

V = len(dic.groupby('1'))
alp = 0.01

ps = []
for ngr in range(1, n+1):
    pps = []
    for par in test:
        pps.append(get_perp(dic, tuple(par.split()), V, alp, ngr))
    print(np.mean(pps))
    ps.append(np.mean(pps))

import matplotlib.pyplot as plt

plt.plot(list(range(1, n+1)), ps)
plt.show()
