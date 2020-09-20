import numpy as np

def gs_coeff(v1, v2):
    return np.dot(v2, v1)/np.dot(v1, v1)

def multiply(coeff, v):
    return map((lambda x: x*coeff), v)

def project(v1, v2):
    return multiply(gs_coeff(v1, v2), v1)


def gs(X):

    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = project(inY, X[i])
            #print "i =", i, ", projection vector =", proj_vec
            temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
            #print "i =", i, ", temporary vector =", temp_vec
        Y.append(temp_vec)

    return Y

test = np.array([[3.0, 1.0], [2.0, 2.0]])

a = gs(test)
print(a)




