from statistics import mean
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

def slope_of_reg(x_val: np.array, y_val: np.array):
    assert isinstance(x_val, np.ndarray)
    assert isinstance(y_val, np.ndarray)
    num = (mean(x_val) * mean(y_val)) - (mean(x_val * y_val))
    dem =( mean(x_val)**2) - (mean(x_val **2))
    m = num/dem
    b = mean(y_val) - m * mean(x_val)
    return m, b

def squared_error(y_orig, y_hat):
    return np.sum((y_orig - y_hat)**2)

def R_squared(y_orig, y_hat):
    y_mean = [mean(y_orig) for y in y_orig]
    squared_error_reg = squared_error(y_orig, y_hat)
    squared_error_mean = squared_error(y_orig, y_mean)
    r_squared = 1 - (squared_error_reg/ squared_error_mean)
    return r_squared

def create_dataset(n_points, variance, step, correlation=False):
    val = 1
    ys = []
    for i in range(n_points):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    x = np.array(xs)
    y = np.array(ys)
    return x, y


x, y = create_dataset(40, 80, 2, 'neg')

m, b = slope_of_reg(x, y)
reg_line = [(m * x) + b for x in x]
r = R_squared(y, reg_line)
print(r)

plt.plot(x, y, 'o')
plt.plot(x, reg_line)
plt.show()