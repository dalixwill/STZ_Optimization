import numpy as np
import os
from Grassmann_ import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colormaps as cmaps


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


def STZ_Darius(x, D0, arg):
    rank0 = (D0.shape[1])
    if arg == 0:
        distance = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            script = '/Users/dariusalix-williams/Documents/Continuum_Comparison/esim/coord_transforms/eran/simple_shear/shear_energy qs tem.0 energy {:7.4f} {:7.4f}'.format(x[i,0],x[i,1])
            os.system(script)
            myArray = np.fromfile('/Users/dariusalix-williams/Documents/Continuum_Comparison/esim/coord_transforms/eran/simple_shear/sct_q.out/tem.24', dtype=np.float32)
            Array = myArray.reshape(36, 71)
            D = Array[1:, 1:]
            print()
            '''
            fig = plt.figure()
            ax = fig.add_subplot(111)
            pos = ax.imshow(D, cmap=cmaps.parula, interpolation='bicubic', \
                        origin='lower', )
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar

            print()
            '''
            u, s, v = svd(D, 0)
            rank = (u.shape[1])
            distance[i] = gr_dist(u, D0, rank, rank0, 'Grassmann')
            print()
    else:
        script = '/Users/dariusalix-williams/Documents/Continuum_Comparison/esim/coord_transforms/eran/simple_shear/shear_energy qs tem.0 energy {:7.4f} {:7.4f}'.format(
            x[0], x[1])
        os.system(script)
        myArray = np.fromfile(
            '/Users/dariusalix-williams/Documents/Continuum_Comparison/esim/coord_transforms/eran/simple_shear/sct_q.out/tem.24',
            dtype=np.float32)
        Array = myArray.reshape(36, 71)
        D = Array[1:, 1:]

        '''
        print()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pos = ax.imshow(D, cmap=cmaps.parula, interpolation='bicubic', \
                        origin='lower', )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar

        print()
        '''

        u, s, v = svd(D, 0)
        rank = (u.shape[1])
        distance = gr_dist(u, D0, rank, rank0, 'Grassmann')

    return distance
