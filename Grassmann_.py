import numpy as np
import scipy
import itertools
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import interp1d


def svd(matrix, value):
    ui, si, vi = np.linalg.svd(matrix, full_matrices=True)
    si = np.diag(si)
    vi = vi.T
    if value == 0:
        rank = np.linalg.matrix_rank(si)
        for i in range(rank):  # increase the number of basis up to rank
            u = ui[:, :i]
            s = si[:i, :i]
            v = vi[:, :i]
    elif value == 1:
        rank = np.linalg.matrix_rank(si, tol=1.0e-4)
        for i in range(201):  # increase the number of basis up to rank
            u = ui[:, :i]
            s = si[:i, :i]
            v = vi[:, :i]
            b = np.dot(u, np.dot(s, v.T))
            norm = np.linalg.norm(matrix - b, 'fro')
            if norm < 1.0e-4:
                ranki = i
                u = ui[:, :ranki]
                s = si[:ranki, :ranki]
                v = vi[:, :ranki]
                break
    elif value == 2:
        rank = np.linalg.matrix_rank(si)
        for i in range(rank):  # increase the number of basis up to rank
            u = ui[:, :i]
            s = si[:i, :i]
            v = vi[:, :i]

    else:
        u = ui[:, :value]
        s = si[:value, :value]
        v = vi[:, :value]

    return u, s, v


def gr_dist(a, b, k,l, arg):
    rank = min(k, l)
    a = a[:,:rank]
    b = b[:,:rank]

    if arg == 'Grassmann':
        r = np.dot(a.T, b)
        ui, si, vi = np.linalg.svd(r, full_matrices=False)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        dist = np.sqrt(abs(k-l)*np.pi**2/4+np.sum(theta ** 2))

    elif arg == 'Chordal':
        r1 = np.dot(a,a.T)
        r2 = np.dot(b, b.T)
        r = r1-r2
        norm1 = 1/np.sqrt(2)*np.linalg.norm(r, 'fro')

        r_star = np.dot(a.T, b)
        ui, si, vi = np.linalg.svd(r_star, full_matrices=False)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        theta = (np.sin(theta))**2
        norm2 = np.sqrt(np.sum(theta))

        dist = np.sqrt(abs(k-l)+np.sum(theta))


    elif arg == 'Procrustes':
        r = np.dot(a.T, b)
        ui, si, vi = np.linalg.svd(r, full_matrices=False)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        theta = np.sin(theta/2)**2
        dist = np.sqrt(abs(k-l)+2*np.sum(theta))


    elif arg == 'projection':
        a = scipy.linalg.orth(a)
        b = scipy.linalg.orth(b)
        # Check rank and swap
        c = np.zeros([a.shape[0], a.shape[1]])
        if a.shape[1] < b.shape[1]:
            a = c
            a = b
            b = c
        # Compute the projection according to[1].
        r = np.dot(a.T, b)
        b = b - np.dot(a, r)

        # Make sure it's magnitude is less than 1.
        dist = np.arcsin(min(1,scipy.linalg.norm(b)))

    return dist
