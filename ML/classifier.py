import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Neural_Net:
    def __init__(self, inputs, output, layers=[5, 4, 3], learning_rate=0.001, iterations=100):
        self.learning_rate = learning_rate
        self.layers = layers
        self.iterations = iterations
        self.X = inputs
        self.y = output
        self.loss = [1, 2, 3]

    def weight_init(self):
        self.w1 = np.random.randn(self.layers[0], self.layers[1])
        self.b1 = np.random.randn(self.layers[1],)
        self.w2 = np.random.randn(self.layers[1], self.layers[2])
        self.b2 = np.random.randn(self.layers[2],)

    def ReLU(self, x):
        return max(0, x.all())

    def Soft_Max(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def cross_entropy(self, y, y_hat):
        n = len(y)
        loss = -1/n * (np.sum(np.multiply(y, np.log(y_hat)) + np.multiply((1-y), np.log(1-y_hat))))
        return loss


    def feedforward(self):
        Z1 = np.dot(self.X, self.w1) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = np.dot(A1, self.w2) + self.b2
        y_hat = self.sigmoid(Z2)
        loss = self.cross_entropy(self.y, y_hat)

        return y_hat, loss, Z1, A1, Z2

    def ReLU_der(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def backpropogation(self, y_hat):
        dl_wrt_yhat = -(np.divide(self.y, y_hat) - np.divide((1 - self.y), (1 - y_hat)))
        dl_wrt_sig = y_hat * (1 - y_hat)
        dl_wrt_z2 = dl_wrt_sig * dl_wrt_yhat

        dl_wrt_A1 = np.dot(dl_wrt_z2, self.w1)
        dl_wrt_w2 = np.dot(dl_wrt_z2, self.feedforward()[3])
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_A1 * self.ReLU_der(self.feedforward()[2])
        dl_wrt_w1 = np.dot(self.X, dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

        self.w1 = self.w1 - self.learning_rate * dl_wrt_w1
        self.w2 = self.w2 - self.learning_rate * dl_wrt_w2
        self.b1 = self.b1 - self.learning_rate * dl_wrt_b1
        self.b2 = self.b2 - self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.weight_init()

        for i in range(self.iterations):
            y_hat, loss, Z1, A1, Z2 = self.feedforward()
            self.backpropogation(y_hat)
            self.loss.append(loss)

    def predict(self, X):
        Z1 = np.dot(self.X, self.w1) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = np.dot(A1, self.w2) + self.b2
        pred = self.sigmoid(Z2)
        return pred

    def accuracy(self, y, y_hat):
        acc = int(sum((y == y_hat)))/(100 * len(y))
        return acc




# headers =  ['age', 'sex','chest_pain','resting_blood_pressure',
#         'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
#         'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak','slope of the peak',
#         'num_of_major_vessels','thal', 'heart_disease']
#
# address = r'C:\Users\ishaa\Downloads\heart.dat'
#
# heart_df = pd.read_csv(address, sep=' ', names=headers)
#
# # heart_df = pd.read_csv('heart_disease.csv')
# X = heart_df.drop(columns=['heart_disease'])
# heart_df['heart_disease'] = heart_df['heart_disease'].replace(1, 0)
# heart_df['heart_disease'] = heart_df['heart_disease'].replace(2, 1)
#
# y_label = heart_df['heart_disease'].values.reshape(X.shape[0], 1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.2, random_state=1)
# sc = StandardScaler()
# sc.fit(X_train)
# sc.transform(X_train)
# sc.transform(X_test)
#
# n_trial = Neural_Net(X_train, y_train, [13, 8, 1])
#
# n_trial.fit(X_train, y_train)

