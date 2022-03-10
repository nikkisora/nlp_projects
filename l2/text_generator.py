
import random

import pandas as pd
import warnings


def get_probas(ngram_df, prec_words):
    """Return dataframe containing words sorted by cumulative sum of probability
    of encountering it after preciding words

    Args:
        ngram_df (DataFrame): dataframe containing all n-grams and number of occurrences
        prec_words (tuple): tuple of preciding words we want to find probabilities for

    Returns:
        DataFrame: DataFrame containing next words and their probabilities
    """
    while prec_words not in ngram_df.index:
        if len(prec_words) == 1:
            prec_words = ('_')
            break
        prec_words = prec_words[1:]
    return ngram_df.loc[prec_words].\
              reset_index()[[str(len(prec_words)+1)]].\
              value_counts(normalize=True).\
              sort_values(ascending=False).\
              cumsum().reset_index()


get_rand_word = lambda df: df[df[0] >= random.random()].iloc[0].iloc[0]


warnings.filterwarnings('ignore')


ngram = 5

dic = pd.read_csv('l2/cyber_dic_7.csv', index_col=[str(i+1) for i in range(ngram)]).sort_index()
print('loaded')

to_gen = 5
gen_len = 300

start = 'It was cold and rainy night'

for _ in range(to_gen):
    print(start, end=' ')
    gen_str = start.split()
    while len(gen_str) < gen_len:
        ng = min(len(gen_str), ngram-1)
        probs = get_probas(dic, tuple(gen_str[-ng:]))
        gen_str.append(get_rand_word(probs))
        print(gen_str[-1], end=' ')
    print('\n\n-----------------------------------------------------')

