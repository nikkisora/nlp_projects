
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

    c = ngram_df.loc[prec_words].groupby(str(len(prec_words)+1)).sum()
    tot = c.sum()
    c = c / tot
    return c.sort_values('num').cumsum().reset_index()


def get_rand_word(df, vocab, alpha):
    """Get random word according to ngram"""
    if random.random() < alpha:
        return vocab.sample(1).iloc[0]
    return df[df['num'] >= random.random()].iloc[0].iloc[0]


warnings.filterwarnings('ignore')

ngram = 5

dic = pd.read_csv('l2/cyber_dic_7.csv', index_col=[str(i+1) for i in range(ngram)]).sort_index()

vocab = dic.groupby('1').sum().reset_index().iloc[:, 0]

to_gen = 2
gen_len = 60
alpha = 0.04

start = 'Elden Ring is'

for _ in range(to_gen):
    gen_str = start.split()
    while len(gen_str) < gen_len:
        ng = min(len(gen_str), ngram-1)
        probs = get_probas(dic, tuple(gen_str[-ng:]))
        gen_str.append(get_rand_word(probs, vocab, alpha))
    print(' '.join(gen_str))
    print('-----------------------------------------------------')

