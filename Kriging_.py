from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor


def predictor(Input, Output):

    kernel = C(1.0, (1e-3, 1e3)) * RBF(5.0, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
    gp.fit(Input, Output)

    return gp






