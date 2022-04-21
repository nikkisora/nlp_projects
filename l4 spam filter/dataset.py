import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def get_dataset(
        dataset_path = './datasets/spam_ham_dataset.csv',
        max_df=0.9,
        min_df=0.01,
    ) -> tuple[pd.Series, pd.Series, set[str], set[str]]:
    """return texts and labels for emails

    Args:
        dataset_path (str, optional): path to dataset csv. Defaults to './datasets/spam_ham_dataset.csv'.
        max_df (float, optional): upper threshold for encountering a word in
        documents, anything higher is cut off. Defaults to 0.9.
        min_df (float, optional): lower threshold for encountering a word in
        documens, anything lower is cut off. Defaults to 0.01.

    Returns:
        tuple[pd.Series, pd.Series, set[str], set[str]]: Texts, labels,
        ham vocab, spam vocab
    """
    spam_ds = pd.read_csv(dataset_path, index_col=0, usecols=[0,2,3])
    spam_ds['text'] = spam_ds['text'].str.lower()

    vectorizer = CountVectorizer(stop_words='english', max_df=max_df, min_df=min_df)
    vectorizer.fit(spam_ds.query('label_num == 0')['text'])
    ham_dict = set(vectorizer.vocabulary_)
    vectorizer.fit(spam_ds.query('label_num == 1')['text'])
    spam_dict = set(vectorizer.vocabulary_)

    return spam_ds['text'], spam_ds['label_num'], ham_dict, spam_dict
