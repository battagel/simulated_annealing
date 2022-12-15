"""
Prove which random method is better
"""


import math
import random
import numpy as np
from matplotlib import pyplot as plt


LENGTH = 46
log1 = []
log2 = []
log3 = []


def generate_random_uni():
    """
    Generate random index with uni method
    """
    # Number between 0 - 1
    random1 = random.random()
    random2 = random.random()

    edges = LENGTH + 1
    index1 = math.floor(random1 / (1/edges))
    if index1 >= edges - 2:
        index1, index2 = generate_random_uni()
    edges_left = LENGTH - index1 - 1
    index2 = math.floor(random2 / (1/(edges_left))) + index1 + 2

    return index1, index2


def generate_random_mine() -> tuple[int, int]:
    """
    Generate two random indexes in size order that are not adjacent
    """
    unacceptable_indexes: bool = True
    random_index1: int = 0
    random_index2: int = 0
    while unacceptable_indexes:
        random_index1 = random.randint(0, LENGTH)
        random_index2 = random.randint(0, LENGTH)
        index_difference: int = abs(random_index1 - random_index2)
        if index_difference > 1:
            unacceptable_indexes = False

    # We DO want to hit the indexes of both 0 and 46 as we want to hit slices
    # with no edges to ensure that the first and large person in ranking is
    # changes.
    # [A-B-C-D]
    # Indexes of 0 and 46 and perform 2-change neighbour
    # [], [A-B-C-D], []
    # = [D-C-B-A]
    # Ranking is reversed which allows 1st and last person to be changed
    smallest_index: int = min([random_index1, random_index2])
    largest_index: int = max([random_index1, random_index2])

    return smallest_index, largest_index


def count_frequency(my_list):
    """
    Counts up the occuranes of values in list and outputs to dict
    """

    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    return freq


def plot():
    """
    Plot graphs
    """

    uni_freq = count_frequency(log1)
    my_freq = count_frequency(log2)

    print("My Average: ")
    print(((sum(log2) / len(log2)) / LENGTH))

    print("Uni Average: ")
    print(((sum(log1) / len(log1)) / LENGTH))

    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].bar(uni_freq.keys(), uni_freq.values())
    ax[1].bar(my_freq.keys(), my_freq.values())

    ax[0].set_title("Uni Random")
    ax[1].set_title("My Random")

    plt.show()


ITERS = 1000000
print("Uni Method")
for _ in range(0, ITERS):
    rand1, rand2 = generate_random_uni()
    log1.append((rand1 + rand2))

print("My Method")
for _ in range(0, ITERS):
    rand1, rand2 = generate_random_mine()
    log2.append((rand1 + rand2))

# print("Control")
# for _ in range(0, ITERS):
#     rand1 = random.random() * LENGTH
#     log3.append(rand1)

plot()
