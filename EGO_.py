import numpy as np
from Kriging_ import predictor
from gfunctions import *
import chaospy as cp
from scipy.stats import norm
from Grassmann_ import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colormaps as cmaps


Tref = np.loadtxt('./tem.24', skiprows=1)
'''
#  Plot Temperature
fig = plt.figure()
ax = fig.add_subplot(111)
pos = ax.imshow(Tref, cmap=cmaps.parula, interpolation='bicubic', \
                origin='lower', )
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar
#plt.show()
'''

u0, s0, v0 = svd(Tref, 0)

dim = 2
N = 21
# define the distribution and the boundaries of beta and U0
beta = cp.Uniform(lo=16, up=18)
U0 = cp.Uniform(lo=-3.6035, up=-3.6021)

# Generate N samples using latin hypercube sampling
X = np.zeros([N, dim])
X[:, 0] = beta.sample(N, rule="L")
X[:, 1] = U0.sample(N, rule="L")

# Calculate the response ffor each pair of (beta, U0)
Y = STZ_Darius(X, u0, 0).ravel()

# Set up the Kriging (Gaussian process) surrogate
gp = predictor(X, Y)

# Generate N_test random realizations of (beta, U0)
N_test = 200
X_test = np.zeros([N_test, dim])
X_test[:, 0] = beta.sample(N_test)
X_test[:, 1] = U0.sample(N_test)

# Use the surrogate in order to estimate the response at the N_test points
y_hat, s2 = gp.predict(X_test, return_std=True)

# Apply the EGO algorithm to find the next sampling point
fmin = min(y_hat)


EI= np.zeros(X_test.shape[0])
for i in range(N_test):
   EI[i] = (fmin-y_hat[i])*norm.cdf((fmin-y_hat[i])/np.sqrt(s2[i]), loc=0.0, scale=1.0) + np.sqrt(s2[i])*norm.pdf((fmin-y_hat[i])/np.sqrt(s2[i]), loc=0.0, scale=1.0)

# Find the position  of the maximum value of EI
J = np.argmax(EI)

'''
x0 = np.sort(X_test[:, 0])
index0 = np.argsort(x0)
x1 = np.sort(X_test[:, 1])
index1 = np.argsort(x1)

plt.figure()
plt.plot(x0, EI[index0], 'k', linewidth=2)

plt.figure()
plt.plot(x1, EI[index1], 'k', linewidth=2)
plt.show()
'''
Iter = 0
while Iter < 100:    # Generate 100 samples
    Iter = Iter +1
    # Estimate the continuum model for the realization that corresponds to the maximum value of EI
    Y_add = STZ_Darius(X_test[J, :], u0, 1).ravel()

    # Add this realization to the initial N points
    Y = np.concatenate((Y, Y_add), axis = 0)
    X = np.concatenate((X, X_test[J, :].reshape(1, dim)))

    # Set up the Kriging model again using also the added point
    gp = predictor(X, Y)

    N_test = 400
    X_test = np.zeros([N_test, dim])
    X_test[:, 0] = beta.sample(N_test, rule="L")
    X_test[:, 1] = U0.sample(N_test, rule="L")

    y_hat, s2 = gp.predict(X_test, return_std=True)
    fmin = min(y_hat)


    EI= np.zeros(X_test.shape[0])
    for i in range(N_test):
        EI[i] = (fmin-y_hat[i])*norm.cdf((fmin-y_hat[i])/np.sqrt(s2[i]), loc=0.0, scale=1.0) + np.sqrt(s2[i])*norm.pdf((fmin-y_hat[i])/np.sqrt(s2[i]), loc=0.0, scale=1.0)

    EIsort = np.sort(EI)
    J = np.argmax(EI)

    print(X_test[J, :])


np.savetxt('{0}.txt'.format('samples'), X)
np.savetxt('{0}.txt'.format('X_test'), X_test)
np.savetxt('{0}.txt'.format('Y_test'), y_hat)
