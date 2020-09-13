'''
This code plots the accuracy of both
datasets with respect to the number of
samples
'''


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from mglearn.datasets import make_wave
from matplotlib import pyplot as plt
import numpy as np


# train_accuracy = []                                   #declaring the score arrays
# test_accuracy = []                                    #for both test and training
# n_samples = range(10, 100, 10)
# for n in n_samples:
#     X, y = make_wave(n_samples=n)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#     reg = KNeighborsRegressor(n_neighbors=3)        #declaring reg as an object with knn = 3
#     reg.fit(X_train, y_train)                       #fitting the model
#     train_accuracy.append(reg.score(X_train, y_train))
#     test_accuracy.append(reg.score(X_test, y_test))
#
# plt.plot(n_samples, train_accuracy, label='Train Accuracy')
# plt.plot(n_samples, test_accuracy, label='Test Accuracy')
# plt.legend()


X, y = make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
plt.suptitle('Nearest Neighbor Regression')
for n_neighbor, ax in zip([1, 3, 5, 7, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbor).fit(X, y)
    ax.plot(X, y, 'o')
    ax.plot(X, -3*np.ones(len(X)), 'o')
    ax.plot(line, reg.predict(line))
    ax.set_title('%d neighbors' % n_neighbor)

plt.show()

