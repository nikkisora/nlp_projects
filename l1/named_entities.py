"""Get named entities from corpus"""

import pandas as pd
import re


df = pd.read_csv('l1/dictionaries/word_freq.csv')

def write_to_file(file_name, lst):
    not_ent = set(df['word']).difference({'for', 'of', 'the'})
    # not_ent = set()
    lst = map(lambda s: s.strip(), lst)
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write('\n'.join(filter(lambda s: s.lower() not in not_ent,lst)))

pop_words = df.query('count > 100000')['word'].dropna()

with open('l1/corpus/corpus_rand.txt', 'r', encoding='utf-8') as f:
    f_t = f.read()
    corp_words = f_t.split()


only_cap = re.sub(r'[^A-Z ]', '', f_t).split()
write_to_file('l1/ne results/only capital.txt', set([s for s in only_cap if len(s)>1]))

possible_names = set(re.findall(r' (?:[A-Z][a-z.]+(?: for | of | of the |[ ./-]))+',f_t))
write_to_file('l1/ne results/capitalized in the middle.txt', possible_names)
write_to_file('l1/ne results/not popular words.txt',
              set(map(lambda x: x.capitalize(), corp_words)).difference(set(map(lambda x: x.capitalize(), pop_words))))
