import pandas as pd
import re
from sklearn import metrics
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm import Laplace, Lidstone
from dataset import get_dataset
from tqdm import tqdm

def clean_text(text):
    """clean text"""
    text = re.sub(r'[^a-z. ]', ' ', text).split()
    sw = set(stopwords.words('English'))
    text = [w for w in text if w not in sw]
    return ' '.join(text)


def get_lm(text, ngram=2, gamma=0.1):
    """generate language model"""
    sents = nltk.sent_tokenize(text)
    word_list = [[w for w in nltk.word_tokenize(sent)] for sent in sents]
    train, vocab = padded_everygram_pipeline(ngram, word_list)
    train = [list(t) for t in train]
    lm = Lidstone(order=ngram, gamma=gamma)
    lm.fit(train, vocab)
    return lm


X, y, _, _ = get_dataset()

X = X.apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    train_size=0.7,
                                                    # test_size=5,
                                                    random_state=2)

X_train_spam = X_train[y_train.values.astype('bool')]
X_train_ham = X_train[~y_train.values.astype('bool')]


ngram = 1
g1 = 0.1
g2 = 0.1

spam_lm = get_lm(' '.join(X_train_spam), ngram=ngram, gamma=g1)
ham_lm = get_lm(' '.join(X_train_ham), ngram=ngram, gamma=g2)

pred = []
for test in tqdm(X_test.values, leave=False):
    sents = nltk.sent_tokenize(test)
    text = [w for sent in sents for w in nltk.word_tokenize(sent)]
    ngr = list(ngrams(pad_both_ends(text, n=ngram), ngram))
    spam_perp = spam_lm.perplexity(ngr)
    ham_perp = ham_lm.perplexity(ngr)
    pred.append(int(spam_perp<ham_perp))

print(f'acc: {metrics.accuracy_score(y_test, pred)}')

print(metrics.classification_report(y_test, pred, target_names=['HAM', 'SPAM']))
print(metrics.confusion_matrix(y_true=y_test, y_pred=pred))