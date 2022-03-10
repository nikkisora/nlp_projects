"""
Get steam reviews for sentiment analisis
"""

import json
from time import sleep

import requests
from requests.adapters import HTTPAdapter, Retry
import tqdm


app_ids = [1621310, # Atelier Sophie 2
           1245620, # Elden ring
           1599340, # Lost ark
           271590,  # GTA V
           1091500, # Cyberpunk 2077
           275850,  # No man's sky
           1517290, # Battlefield 2042
           730,570,578080,1172470,252490,440,359550,346110,1794680,
           1506830,230410,381210]


reviews = []

# URL = f'https://store.steampowered.com/appreviews/{app_ids[-1]}'
S = requests.Session()
retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[ 500, 502, 503, 504 ])
S.mount('http://', HTTPAdapter(max_retries=retries))
params = {
    'json': '1',
    'purchase_type': 'all',
    'filter': 'recent',
    'num_per_page':'100',
    'cursor': '*'
}

p_bar = tqdm.tqdm(position=1)

for app in tqdm.tqdm(app_ids, position=0):
    url = f'https://store.steampowered.com/appreviews/{app}'
    params['cursor'] = '*'
    t = 0
    while 1:
        # data = get_data(params, url, S)
        resp = requests.get(url, params)
        if resp.status_code != 200:
            if t == 5:
                break
            t += 1
            sleep(5)
            continue
        data = resp.json()
        if not data['reviews']:
            break
        for review in data['reviews']:
            reviews.append({'review':review['review'],
                            'positive':review['voted_up']})
        p_bar.update(100)
        params['cursor'] = data['cursor']

p_bar.close()

with open("steam_reviews/reviews_corp.json", "w") as outfile:
    json.dump(reviews, outfile)
