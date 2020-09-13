import numpy as np
import pandas as pd
from collections import Counter
import warnings
import random

def knn_classifier(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than the voting groups')

    distances = []
    for group in data:
        for feature in data[group]:
            euclid_dist = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([euclid_dist, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result, confidence

address = r'C:\Users\ishaa\Downloads\breast-cancer-wisconsin.data'
df = pd.read_csv(address, na_values='?', header=0).fillna(-9999).drop('id', 1)

full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

'''
[i[-1]] is used to identify the class
Appending all the features upto the last element
last element is the class
'''
[train_set[i[-1]].append(i[:-1]) for i in train_data]
[test_set[i[-1]].append(i[:-1]) for i in test_data]


correct = 0
total  = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = knn_classifier(train_set, data, 5)
        if group == vote:
            correct += 1
        else:
            print(confidence)
        total += 1

accuracy = correct/total
print(accuracy)
