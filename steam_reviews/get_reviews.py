"""
Get steam reviews for sentiment analisis
"""

import json

import requests
import tqdm


def get_data(params, url, session):
    '''Return JSON from get request'''
    return session.get(url=url, params=params).json()


app_ids = [1245620, # Elden ring
           1599340, # Lost ark
           271590,  # GTA V
           1091500, # Cyberpunk 2077
           275850,  # No man's sky
           1517290, # Battlefield 2042
           1621310] # Atelier Sophie 2

reviews = []

URL = f'https://store.steampowered.com/appreviews/{app_ids[-1]}'
S = requests.Session()
params = {
    'json': '1',
    'purchase_type': 'all',
    'filter': 'recent',
    'num_per_page':'100',
    'cursor': '*'
}

p_bar = tqdm.tqdm()

while 1:
    data = get_data(params, URL, S)
    if not data['reviews']:
        break
    for review in data['reviews']:
        reviews.append({'review':review['review'],
                        'positive':review['voted_up']})
    p_bar.update(100)
    params['cursor'] = data['cursor']

p_bar.close()

with open("steam_reviews/reviews_sophie2.json", "w") as outfile:
    json.dump(reviews, outfile)
