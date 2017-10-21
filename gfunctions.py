import numpy as np


def f(x):
    return x * np.sin(x)


def f2(x):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = x[i][0]*np.sin(x[i][1])

    return np.array(y)


def branin_hoo(x, arg):
    a = 1
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)

    if arg == 0:

        y = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            y[i] = a*(x[i, 1] - b*x[i, 0]**2 + c*x[i, 0] - r)**2 + s*(1-t)*np.cos(x[i, 0]) + s +5*x[i, 0]

    else:
        y = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s +5*x[0]

    return y


'''
def STZ_Darius(x, D0):
    rank0 = (u0.shape[1])
    D : Displacement filed from Darius code
    u, s, v = svd(D, 0)
    rank = (u.shape[1])
    distance = gr_dist(u, u0, rank, rank0, 'Grassmann')
    
    return distance
    
'''