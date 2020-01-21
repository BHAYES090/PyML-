import numpy as np
from math import sqrt
import warnings
from collections import Counter
from collections import defaultdict
import pandas as pd
import random

## below to K nearest Neighbors algorithim is defined]

def k_nearest_neighbors(data, predict, k = 3):
    if len(data) >= k:
        warnings.warm('K is set to a value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            #alternate version (harder to comprehend) provided by numpy to calculte euclidean distance
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence

df = pd.read_csv("HayesRothData")
df.replace('?', -99999, inplace = True)
df.drop(['Name'], 1, inplace = True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.99
train_set = {defaultdict(list)}
test_set = {defaultdict(list)}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy:', correct/total * 100.0)
accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))
