'''
This code is to visualise the decision boundary
between two classes as the number of neighbors
increases
'''

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

cancer = load_breast_cancer()
print(cancer.keys())
# print(cancer['data'].shape)
# print(cancer['target_names'])
X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'], random_state=0)
# print(cancer['data'].shape)
# print(cancer['target'].shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

train_accuracy = []
test_accuracy = []

neighbor_settings = range(1, 11)

for n_neighbors in neighbor_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    train_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbor_settings, train_accuracy, label='Training Set')
plt.plot(neighbor_settings, test_accuracy, label='Test Set')
plt.xlabel('# of neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

