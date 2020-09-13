from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

'''
The iris dataset is loaded on to the var iris which is called Bunched. Works like a dictionary
X_train, y_train are the training dataset while X_test, y_test are the test datasets to evaluate model accuracy
knn is the object of class KNeighborsClassifier, # of nearest neighbors is set to 1

'''

iris = load_iris()
# print(iris['data'][:5])
# print(iris['target'])

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
print(iris['data'].shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

fig, ax = plt.subplots(3, 3, figsize=(15, 15))
plt.suptitle("iris_pairplot")
for i in range(3):
    for j in range(3):
        ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
        ax[i, j].set_xticks(())
        ax[i, j].set_yticks(())
        if i == 2:
            ax[i, j].set_xlabel(iris['feature_names'][j])
        if j == 0:
            ax[i, j].set_ylabel(iris['feature_names'][i + 1])
        if j > i:
            ax[i, j].set_visible(False)

# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
#
# X_new = np.array([[5, 2.9, 1, 0.2]])
# X_new.reshape(1, 4)
# print(ax)

# prediction = knn.predict(X_new)
# print(prediction)
# print(iris['target_names'][prediction])
# y_pred = knn.predict(X_test)
# print(knn.score(X_test, y_test))
# print(iris['target_names'][y_pred])
# plt.show()


