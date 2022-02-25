"""main module"""

import json

import pandas as pd

from common import lev_d, jac_d


def spell(str1, dic, ret=5, dist=0.8, steps=0):
    """Return a table of words sorted by distance to the input string

    Args:
        str1 (str): input string
        dic (dict): Dictionary of words created by generate_dictionary.py
        ret (int, optional): Number of rows in the return table. Defaults to 5.
        dist (float, optional): Normalized allowed distance between hashes computed by Jaccard distance. Defaults to 0.8.
        steps (int, optional): Allowed distance in number of different characters in an input string. Defaults to 0.

    Returns:
        DataFrame: A table containing two rows word and distance sorted by distance
    """
    if steps:
        dist = 1 - steps/len(str1)
    str1 = str1.lower()
    table = {'word':[],
            'distance':[]}
    for key in dic:
        if jac_d(key, str1) > dist:
            for word in dic[key]:
                table['word'].append(word)
                table['distance'].append(lev_d(str1, word))
    df = pd.DataFrame(table)
    return df.sort_values('distance', ascending=False).head(ret)

def spell2(str1, dic, ret=5, dist=0.8, steps=0, freq_imp = 0.2):
    """Return a table of words sorted by distance to the input string

    Args:
        str1 (str): input string
        dic (dict): Dictionary of words created by generate_dictionary.py
        ret (int, optional): Number of rows in the return table. Defaults to 5.
        dist (float, optional): Normalized allowed distance between hashes computed by Jaccard distance. Defaults to 0.8.
        steps (int, optional): Allowed distance in number of different characters in an input string. Defaults to 0.
        freq_imp (float, optional): How much word frequency affects distance. Defaults to 0.2.

    Returns:
        DataFrame: A table containing two rows word and distance sorted by distance
    """
    if steps:
        dist = 1 - steps/len(str1)
    str1 = str1.lower()
    table = {'word':[],
            'distance':[]}
    for key in dic:
        if jac_d(key, str1) > dist:
            for word in dic[key]:
                table['word'].append(word[0])
                table['distance'].append((1-freq_imp)*lev_d(str1, word[0])[0]
                                         +freq_imp*word[1])
    df = pd.DataFrame(table)
    return df.sort_values('distance', ascending=False).head(ret)


if __name__ == '__main__':
    with open('l1/words_freq_dict.json', 'r') as json_file:
        word_dict = json.load(json_file)
    while 1:
        try:
            inp = input()
            print(spell2(inp, word_dict, steps=2, freq_imp=0.5))
        except KeyboardInterrupt:
            break
