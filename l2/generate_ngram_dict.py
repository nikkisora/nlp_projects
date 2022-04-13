"""Generate dictionary of n-grams"""
import re
from os import walk

import pandas as pd


df = pd.read_json('steam_reviews/reviews_elden_ring.json')
full_text = ' '.join(df['review'])

# dir_to_check = 'l2/books/'
# filenames = next(walk(dir_to_check), (None, None, []))[2]
# full_text = ''
# for file_name in filenames:
#     with open(dir_to_check+file_name) as file:
#         full_text += file.read()

full_text = re.sub(r'\[.*\]', ' ', full_text) # remove text styling
full_text = re.sub(r'[^A-Za-z!?;\'. ]', ' ', full_text) # remove unneeded characters
full_text = re.sub(r' +', ' ', full_text) # remove multi-spaces
full_text = re.sub(r'[!?;.]+', ' . _', full_text) # Change all sentence endings to single period and add _ as sentence begining

word_list = full_text.split()

n_grams = []
n = 7

for k in range(len(word_list)-n+1):
    n_grams.append(tuple(word_list[k:k+n]))

# n_grams = set(n_grams)

df = pd.DataFrame(n_grams, columns=[str(i+1) for i in range(n)])
# df.fillna(' ', inplace=True)


get_n_grams = lambda df, n: df.groupby(df.columns.tolist()[:n]).\
                              size().\
                              reset_index().\
                              rename(columns={0:'num'})

dic = get_n_grams(df, n).set_index([str(i+1) for i in range(n-1)]).sort_index()

dic.to_csv('l2/elden_dict.csv')
