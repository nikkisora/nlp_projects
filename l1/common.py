"""Common functions"""

import numpy as np


def lev_d(str1, str2):
    """Return Levenshtein distance between two strings
    Returns:
        tuple: (normalized distance, distance in steps)
    """
    # Initialize starting matrix
    rows = len(str1)+1
    cols = len(str2)+1
    distance = np.zeros((rows, cols), dtype=int)

    for row in range(1, rows):
        distance[row][0] = row
    for col in range(1, cols):
        distance[0][col] = col

    # Calculate distance matrix
    for col in range(1, cols):
        for row in range(1, rows):
            cost = str1[row-1] != str2[col-1]
            distance[row][col] = min(distance[row-1][col] + 1,
                                     distance[row][col-1] + 1,
                                     distance[row-1][col-1] + cost)
    print(distance)
    ratio = 1 - distance[row][col] / max(len(str1), len(str2))
    return round(ratio, 5), distance[row][col]

def d_lev_d(str1, str2):
    """Return Demerau-Levenshtein distance between two strings
    Returns:
        tuple: (normalized distance, distance in steps)
    """
    # Initialize starting matrix
    rows = len(str1)+1
    cols = len(str2)+1
    distance = np.zeros((rows, cols), dtype=int)

    for row in range(1, rows):
        distance[row][0] = row
    for col in range(1, cols):
        distance[0][col] = col

    # Calculate distance matrix
    for col in range(1, cols):
        for row in range(1, rows):
            cost = str1[row-1] != str2[col-1]
            distance[row][col] = min(distance[row-1][col] + 1,
                                     distance[row][col-1] + 1,
                                     distance[row-1][col-1] + cost)
            # Check for transposition
            if (col > 1 and row > 1 and
                str1[row-1] == str2[col-2] and
                str1[row-2] == str2[col-1]):
                distance[row][col] = min(distance[row][col],
                                         distance[row-2][col-2] + 1)

    ratio = 1 - distance[row][col] / max(len(str1), len(str2))
    return round(ratio, 5), distance[row][col]

def jac_d(str1, str2):
    """Return Jaccard distance between two strings"""
    str1_s = set(str1)
    str2_s = set(str2)
    return len(str1_s & str2_s) / len(str1_s | str2_s)


def str_hash(str1):
    """Return unique sorted characters of a string"""
    return ''.join(sorted(set(str(str1))))
