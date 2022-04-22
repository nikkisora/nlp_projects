#%%
from sklearn.decomposition import TruncatedSVD
import fasttext
import numpy as np
import pandas as pd
from get_dataset import get_dataset
import PMI_matrix
import hdbscan

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

dataset, vocab = get_dataset(files_dir='D:/code/nlp_projects/l2/books',
                             remove_stopwords=True,
                             stem=False,
                             remove_singles = 2,
                             chunk_size=250)



# LSA on PPMI

mat, t2i, i2t = PMI_matrix.get_matrix(dataset, window=11)
svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42, algorithm='arpack')
trunk_mat = svd.fit_transform(mat)
#normalize
X_lsa = trunk_mat / np.sqrt(np.sum(trunk_mat*trunk_mat, axis=1, keepdims=True))

clusterer_lsa = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1, cluster_selection_method='leaf')
cluster_lsa_labels = clusterer_lsa.fit_predict(X_lsa)
print('LSA number of clusters: ', len(set(cluster_lsa_labels)))


# Fasttext

dataset_string = [' '.join(doc) for doc in dataset]
dataset_string = '\n'.join(dataset_string)

with open('D:/code/nlp_projects/l3/fasttext_dataset.txt', 'w+') as file:
    file.write(dataset_string)

ft_model = fasttext.train_unsupervised('D:/code/nlp_projects/l3/fasttext_dataset.txt',
                                       'skipgram')

X = []
for w in range(len(vocab)):
    X.append(ft_model[i2t[w]])
X_ft = np.array(X)

clusterer_ft = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1, cluster_selection_method='leaf')
cluster_ft_labels = clusterer_ft.fit_predict(X_ft)
print('Fasttext number of clusters: ', len(set(cluster_ft_labels)))


# graphs

projection_lsa = TSNE().fit_transform(X_lsa)
projection_ft = TSNE().fit_transform(X_ft)

fig, (ax_lsa, ax_ft) = plt.subplots(1, 2, sharey=True)

color_palette_lsa = sns.color_palette('Paired', len(set(cluster_lsa_labels)))
cluster_colors_lsa = [color_palette_lsa[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer_lsa.labels_]
cluster_member_colors_lsa = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors_lsa, clusterer_lsa.probabilities_)]
ax_lsa.scatter(*projection_lsa.T, s=50, linewidth=0,
               c=cluster_member_colors_lsa, alpha=0.25)

color_palette_ft = sns.color_palette('Paired', len(set(cluster_ft_labels)))
cluster_colors_ft = [color_palette_ft[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer_ft.labels_]
cluster_member_colors_ft = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors_ft, clusterer_ft.probabilities_)]
ax_ft.scatter(*projection_ft.T, s=50, linewidth=0,
               c=cluster_member_colors_ft, alpha=0.25)

plt.show()

# save dataframes with clusters
#%%
df_lsa = pd.DataFrame({'cluster':cluster_lsa_labels}).reset_index()
df_lsa['index'] = df_lsa['index'].apply(lambda x: i2t[x])
print('LSA cluster sizes:\n', df_lsa['cluster'].value_counts())

words_in_cluster = dict()
for i in range(-1,len(set(cluster_lsa_labels))-1):
    words_in_cluster[f'{i}'] = list(df_lsa.query(f'cluster == {i}').iloc[:,0])

words_in_cluster = pd.DataFrame.from_dict(words_in_cluster, orient='index')
words_in_cluster = words_in_cluster.transpose()
words_in_cluster.to_csv('D:/code/nlp_projects/l3/results/lsa_clusters_1.csv')


df_ft = pd.DataFrame({'cluster':cluster_ft_labels}).reset_index()
df_ft['index'] = df_ft['index'].apply(lambda x: i2t[x])
print('Fasttext cluster sizes:\n', df_ft['cluster'].value_counts())


words_in_cluster = dict()
for i in range(-1,len(set(cluster_ft_labels))-1):
    words_in_cluster[f'{i}'] = list(df_ft.query(f'cluster == {i}').iloc[:,0])

words_in_cluster = pd.DataFrame.from_dict(words_in_cluster, orient='index')
words_in_cluster = words_in_cluster.transpose()
words_in_cluster.to_csv('D:/code/nlp_projects/l3/results/fasttext_clusters_1.csv')



# %%
