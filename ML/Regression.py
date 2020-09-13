import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import datetime
from matplotlib import pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Low'])/(df['Adj. Low'])) * 100
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open'])/(df['Adj. Open'])) * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

forecast_out = math.ceil(0.01 * len(df))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
y = y[:-forecast_out]
# print(len(X), len(X_lately), len(y), len(df))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2,random_state=1)
clf = LinearRegression(n_jobs=1)
clf.fit(X_train, y_train)
preds = clf.predict(X_lately)
df['Forecastset'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


for i in preds:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for a in range(len(df.columns)-1)] + [i]

df['Forecastset'].plot()
df['Adj. Close'].plot()
plt.legend(loc=4)
print(df)


plt.show()
