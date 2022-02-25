"""
    get_category_items.py

    MediaWiki API Demos
    Demo of `Categorymembers` module : List twenty items in a category

    MIT License
"""

import re

import requests
import tqdm


def get_data(params, url, session):
    '''Return JSON from get request'''
    return session.get(url=url, params=params).json()


categories = ['American computer scientists',
              'Genomics',
              'Genetics',
              'Game theory',
              'Renaissance',
              'Early Modern period']
pages = []

URL = "https://en.wikipedia.org/w/api.php"
S = requests.Session()
param = {
    "action": "query",
    "cmlimit": "500",
    "list": "categorymembers",
    "format": "json"
}


for cat in categories:
    param['cmtitle'] = f'Category:{cat}'
    data = get_data(param, URL, S)
    PAGES = data['query']['categorymembers']
    pages.extend([p['title'] for p in PAGES])
    while 'continue' in data:
        data = get_data(param | data['continue'], URL, S)
        PAGES = data['query']['categorymembers']
        pages.extend([p['title'] for p in PAGES])

with open('l1/corpus/corpus.txt', 'w+', encoding='utf-8') as corpus:
    for t in tqdm.tqdm(pages):
        if 'Category:' in t or 'List of' in t:
            continue

        param = {
            "action": "query",
            "prop": "extracts",
            "titles": t,
            "explaintext": "1",
            "formatversion": "2",
            "format": "json"
        }

        data = get_data(param, URL, S)
        for d in data['query']['pages']:
            f_name = f'l1/articles/{re.sub(r"[^a-zA-Z ]", "", d["title"])}.txt'
            with open(f_name, 'w', encoding='utf-8') as f:
                f.write(d['extract'])
            corpus.write(d['extract'])
