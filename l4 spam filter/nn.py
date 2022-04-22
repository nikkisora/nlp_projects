#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os


import re
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics


from dataset import get_dataset



X, y, _, _ = get_dataset('D:/code/nlp_projects/datasets/spam_ham_dataset.csv', min_df=0)
X = X.apply(lambda s: re.sub(r'\.', '', s))





import nltk
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

tokenize = lambda x: word_tokenize(x)

ps = PorterStemmer()
stem = lambda w: [ ps.stem(x) for x in w ]

lemmatizer = WordNetLemmatizer()
leammtizer = lambda x: [ lemmatizer.lemmatize(word) for word in x ]

X = X.apply(tokenize)
X = X.apply(stem)
X = X.apply(leammtizer)
X = X.apply(lambda x: ' '.join(x))


# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
#                                                     stratify=y, random_state=2)


max_words = 10000
cv = CountVectorizer(max_features=max_words, stop_words='english')
sparse_matrix = cv.fit_transform(X).toarray()
X_train, X_test, y_train, y_test = train_test_split(sparse_matrix, y, train_size=0.7,
                                                    stratify=y, random_state=2)

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(10000, 100)
        self.linear2 = nn.Linear(100, 10)
        self.linear3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


model = LogisticRegression()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters() , lr=0.01)

x_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train.values)).long()


epochs = 20
model.train()
loss_values = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss_values.append(loss.item())
    pred = torch.max(y_pred, 1)[1].eq(y_train).sum()
    acc = pred * 100.0 / len(x_train)
    print('Epoch: {}, Loss: {}, Accuracy: {}%'.format(epoch+1, loss.item(), acc.numpy()))
    loss.backward()
    optimizer.step()


x_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test.values)).long()


#%%
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    loss = criterion(y_pred, y_test)
    pred = torch.max(y_pred, 1)[1].eq(y_test).sum()
    y_pred_1 = torch.max(y_pred, 1)[1]
    print(metrics.classification_report(y_test, y_pred_1, target_names=['HAM', 'SPAM']))
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_1))
    print ("Accuracy : {}%".format(100*pred/len(x_test)))
# %%
