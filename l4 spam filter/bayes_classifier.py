

import numpy as np
import pandas as pd
from dataset import get_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


X, y, ham_v, spam_v = get_dataset(min_df=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                    stratify=y)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

parameters = {'fit_prior':(False, True), 'alpha':np.arange(0.05,1,0.05)}
nb_cl = MultinomialNB()

clf = GridSearchCV(nb_cl, parameters, cv=3, n_jobs=-1)

clf.fit(tfidf_train, y_train)

pred = clf.predict(tfidf_test)

print(clf.best_params_)

print(metrics.classification_report(y_test, pred, target_names=['HAM', 'SPAM']))
print(metrics.confusion_matrix(y_true=y_test, y_pred=pred))



