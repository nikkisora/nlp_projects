

from tqdm import tqdm
from typing import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def ww_sim(word, mat, tok2indx, indx2tok, topn=10):
    """Calculate topn most similar words to word"""
    indx = tok2indx[word]
    if isinstance(mat, csr_matrix):
        v1 = mat.getrow(indx)
    sims = cosine_similarity(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    return sim_word_scores


def get_matrix(dataset):
    # count all words
    w_counts = Counter()
    for doc in tqdm(dataset):
        for w in doc:
            w_counts[w] += 1

    tok2indx = {tok: indx for indx, tok in enumerate(w_counts.keys())}
    indx2tok = {indx: tok for tok,indx in tok2indx.items()}

    # count skipgrams
    window = 4
    skipgram_counts = Counter()
    for doc in tqdm(dataset):
        tokens = [tok2indx[tok] for tok in doc]
        for word_index, _ in enumerate(tokens):
            context_min = max(0, word_index - window)
            context_max = min(len(doc) - 1, word_index + window)
            contexts = [i for i in range(context_min, context_max + 1)
                        if i != word_index]
            for context_i in contexts:
                skipgram = (tokens[word_index], tokens[context_i])
                skipgram_counts[skipgram] += 1

    rows = []
    cols = []
    values = []
    for (tok1, tok2), sg_count in skipgram_counts.items():
        rows.append(tok1)
        cols.append(tok2)
        values.append(sg_count)
    ww_mat = csr_matrix((values, (rows, cols)))

    # PPMI calculation
    num_skipgrams = ww_mat.sum()
    rows = []
    cols = []
    ppmi_vals = []

    sum_over_words = np.array(ww_mat.sum(axis=0)).flatten()
    sum_over_contexts = np.array(ww_mat.sum(axis=1)).flatten()

    #smoothing
    alpha = 0.75
    sum_over_words_alpha = sum_over_words**alpha
    nca_denom = np.sum(sum_over_words_alpha)

    for (tok_word, tok_context), sg_count in tqdm(skipgram_counts.items()):
        nwc = sg_count
        Pwc = nwc / num_skipgrams

        nw = sum_over_contexts[tok_word]
        Pw = nw / num_skipgrams

        nca = sum_over_words_alpha[tok_context]
        Pca = nca / nca_denom

        ppmi = max(np.log2(Pwc/(Pw*Pca)), 0)

        rows.append(tok_word)
        cols.append(tok_context)
        ppmi_vals.append(ppmi)

    ppmi_mat = csr_matrix((ppmi_vals, (rows, cols)))

    return ppmi_mat, tok2indx, indx2tok