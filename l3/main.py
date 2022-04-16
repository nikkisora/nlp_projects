#%%
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import fasttext
import numpy as np
import pandas as pd
from get_dataset import get_dataset
import PMI_matrix


dataset, vocab = get_dataset(files_dir='D:/code/nlp_projects/l2/books', chunk_size=250)

mat, t2i, i2t = PMI_matrix.get_matrix(dataset)

svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
trunk_mat = svd.fit_transform(mat)
#%%
df = pd.DataFrame(trunk_mat).reset_index()

#%%
df['index'] = df['index'].apply(lambda x: i2t[x])


# %%
from sklearn.cluster import DBSCAN
X = df.iloc[:,1:].values
lsa_clust = DBSCAN().fit(X)

core_samples_mask = np.zeros_like(lsa_clust.labels_, dtype=bool)
core_samples_mask[lsa_clust.core_sample_indices_] = True
labels = lsa_clust.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 0]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
        lw=0,
        alpha=0.01
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
        lw=0,
        alpha=0.01
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()
# %%
