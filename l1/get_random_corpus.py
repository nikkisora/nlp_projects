"""Get random articles from wikipedia"""

import requests
import tqdm
import re

num = 10000
min_symbols = 2000

URL = "https://en.wikipedia.org/w/api.php"
PARAMS = {
    'action': 'query',
    'format':'json',
    'generator':'random',
    'grnnamespace':'0',
    'prop':'extracts',
    'explaintext':'1',
    'formatversion':'2'
}
S = requests.Session()

pages = 0

with open('l1/corpus_rand.txt', 'w+', encoding='utf-8') as corpus:
    with tqdm.tqdm(total=num) as p_bar:
        while pages < num:
            R = S.get(url=URL, params=PARAMS)
            data = R.json()['query']['pages'][0]
            if len(data['extract']) < min_symbols:
                continue
            with open(f'l1/rand_articles/{re.sub(r"[^a-zA-Z ]", "", data["title"])}.txt', 'w', encoding='utf-8') as f:
                f.write(data['extract'])
            corpus.write(data['extract'])
            pages += 1
            p_bar.update(1)

