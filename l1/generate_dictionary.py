"""Generate dictionary for spellchecker"""

import re
import json

import tqdm

from common import str_hash


with open('l1/corpus.txt', 'r', encoding='utf-8') as f:
    f_t = f.read().lower()
    f_t = re.sub(r'[-/]+', ' ', f_t)
    f_t = re.sub(r'[^a-z ]','',f_t)

    d = dict()
    for w in tqdm.tqdm(set(f_t.split())):
        hs = str_hash(w)
        if hs in d:
            if w not in d[hs]:
                d[hs].append(w)
        else:
            d[hs] = [w]
    with open("l1/words.json", "w") as outfile:
        json.dump(d, outfile)

# import pandas as pd

# df = pd.read_csv('l1/word_freq.csv', na_values='not number')
# df['rel_freq'] = 1 - df.index/df['word'].count()

# d = dict()
# for _, r in df.iterrows():
#     hs = str_hash(r['word'])
#     if hs in d:
#         if r['word'] not in d[hs]:
#             d[hs].append((r['word'], r['rel_freq']))
#     else:
#         d[hs] = [(r['word'], r['rel_freq'])]

# with open("l1/words_freq_dict.json", "w") as outfile:
#     json.dump(d, outfile)