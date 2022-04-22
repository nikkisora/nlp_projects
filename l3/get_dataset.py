

import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from collections import Counter

def get_dataset(files_dir, remove_stopwords=True,
                stem = True, remove_singles = 1,
                chunk_size=100):
    """get dataset"""
    files = Path(files_dir).iterdir()
    corp = ''

    for f in files:
        with open(f) as file:
            corp += file.read().lower()

    corp = re.sub(r'[^A-Za-z\n ]', ' ', corp)
    corp = re.sub(r'  ', ' ', corp).split()
    # remove stopwords
    if remove_stopwords:
        sw = set(stopwords.words('English'))
        corp = [w for w in corp if w not in sw]
    # stemming
    if stem:
        ps = PorterStemmer()
        corp = [ps.stem(w) for w in corp]

    if remove_singles:
        counted = Counter(corp)
        corp = [w for w in corp if counted[w] > remove_singles]


    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    return list(chunks(corp, chunk_size)), set(corp)