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


def gr_measure(DIM, simplex,   max_rank, case, scale, Distance):
    psi, sigma, phi, max_rank, rank = project_points(simplex,'old', max_rank, case)
    l = len(psi)
    dist = 0.0
    metr = 0

    if DIM == 2:
        li = np.zeros(3)
    elif DIM == 3:
        li = np.zeros(6)

    for combination in itertools.combinations(range(l), 2):
        li[metr] = gr_dist(psi[combination[0]], psi[combination[1]], rank[combination[0]],rank[combination[1]], Distance)
        dist = dist + li[metr]
        metr = metr+ 1

    if scale == 'no':
        distw = dist
    else:
        distw = dist*simplex.volume

    return dist, distw, li


def project_points(simplex, value, max_rank, case):
    f = np.array([p.functionValue for p in simplex.points])
    l = f.shape[0]  # number of points  Gr(n, k)
    rank = np.zeros(l, dtype=np.int32)  # rank of each point
    # Perform full svd in order to find the rank of each point
    if value == 'old':
        for j in range(l):
            u, s, v = svd(f[j], 0)
            rank[j] = u.shape[1]

        #max_rank = int(np.min(rank))  # rank = max(r1, r2, ..., rn)
        psi = []  # initialize the left singular values as a list
        sigma = []  # initialize the singular values as a list
        phi = []  # initialize the right singular values as a list

        # For each point perform svd with max_rank columns
        for i in range(l):
            if case == 1:
                u, s, v = svd(f[i], rank[i])
            else:
                u, s, v = svd(f[i], max_rank)
            psi.append(u)
            sigma.append(s)
            phi.append(v)

    return psi, sigma, phi, max_rank, rank


def project_tangent(u):
    psi0 = u[0]  # reference point where the tangent is estimated
    n = psi0.shape[0]
    nr = len(u) - 1  # number of points
    gamma = []
    for j in range(nr):
        m = (np.dot(np.dot((np.eye(n) - np.dot(psi0, psi0.T)), u[j+1]), np.linalg.inv(np.dot(psi0.T, u[j+1]))))
        ui, si, vi = np.linalg.svd(m, full_matrices=False)              #svd(m, max_rank)
        gamma.append(np.dot(np.dot(ui, np.diag(np.arctan(si))), vi))
        print()
    return gamma

def interpolate_points(point, nodes, matrix):
    dim = nodes.shape[1]
    input = nodes
    if dim == 1:
        input = input[:,0]

    n = matrix[0].shape[0]
    r = matrix[0].shape[1]
    interp_matrix = np.zeros([n, r])
    origin = np.zeros([n, r])
    output = np.zeros(input.shape[0])
    for j in range(n):
        for k in range(r):
            for l in range(input.shape[0]):
                if l == 0:
                    output[l] = origin[j, k]
                else:
                    output[l] = matrix[l-1][j, k]
            if dim == 1:
                myInterpolator = interp1d(input, output)
            else:
                myInterpolator = NearestNDInterpolator(input, output)

            interp_matrix[j, k] = myInterpolator(point)
            print()
    return interp_matrix


def interpolate_sigma(point, nodes, matrix):
    dim = nodes.shape[1]
    input = nodes
    if dim == 1:
        input = input[:, 0]
    l = len(matrix)
    sigma = []
    for i in range(l):
        sigma.append(np.diag(matrix[i]))

    n = sigma[0].shape[0]
    output = np.zeros(l)
    interp_matrix = np.zeros(n)
    for j in range(n):
        for k in range(l):
            output[k] = sigma[k][j]
        if dim == 1:
            myInterpolator = interp1d(input, output)
        else:
            myInterpolator = NearestNDInterpolator(input, output)
        interp_matrix[j] = myInterpolator(point)
    print()

    return np.diag(interp_matrix)

def calculate_basis(point, nodes, matrix, a):
    interp_matrix = interpolate_points(point, nodes, matrix)
    ui, si, vi = np.linalg.svd(interp_matrix, full_matrices=False)
    X = np.dot(np.dot(np.dot(a, vi.T), np.diag(np.cos(si))) + np.dot(ui, np.diag(np.sin(si))), vi)
    print()
    return X


def project_basis(simplex, nodes, dimension, xi, max_rank, Dpl, case, Distance):

    psi, sigma, phi, max_rank, rank = project_points(simplex, 'old', max_rank, case)   # Perform svd on the solution function and find U, Σ, V a.k.a Ψ, Σ, Φ
    tanpsi = project_tangent(psi)    # matrix Γ on the tangent space for Ψ
    tanphi = project_tangent(phi)    # matrix Γ on the tangent space for Φ

    # sample inside the current simplex
    number_samples = 1
    if xi == 'None' and Dpl == 'None':
        if dimension == 1:
            xi = SampleInSimplex1D(nodes, number_samples, None, dimension)
        else:
            xi = SampleInSimplex(nodes, number_samples, None)

        error = np.zeros(number_samples)
        for i in range(number_samples):
            if dimension == 1:
                sample = xi[0][i]
            else:
                sample = xi[i, :]

            approx_psi = calculate_basis(sample, nodes, tanpsi, psi[0])
            approx_phi = calculate_basis(sample, nodes, tanphi, phi[0])
            approx_sigma = interpolate_sigma(sample, nodes, sigma)
            F_tilde = approximate_model(approx_psi, approx_sigma, approx_phi)
            sample_ = transform_STZ(dimension, sample, 1)
            script = "./shear_test2 qs"
            write_file(dimension, sample_)
            Dpl = gfunSZT(script, 1)

            Psi, Sigma, Phi = svd(Dpl, max_rank)
            k = approx_psi.shape[1]
            l = Psi.shape[1]
            dist = gr_dist(Psi, approx_psi, k, l, Distance)


        #Error = 1/number_samples*np.sum(error)
        output = []
        output.append(np.sqrt(dist / max(k, l)))
        output.append(sample)
        output.append(F_tilde)
        output.append(Dpl)

    else:
        for i in range(number_samples):
            if dimension == 1:
                sample = xi[0][i]
            else:
                sample = xi

            approx_psi = calculate_basis(sample, nodes, tanpsi, psi[0])
            approx_phi = calculate_basis(sample, nodes, tanphi, phi[0])
            approx_sigma = interpolate_sigma(sample, nodes, sigma)
            F_tilde = approximate_model(approx_psi, approx_sigma, approx_phi)
            Psi, Sigma, Phi = svd(Dpl, max_rank)
            k = approx_psi.shape[1]
            l = Psi.shape[1]
            dist = gr_dist(Psi, approx_psi, k, l, Distance)

        output = np.sqrt(dist / max(k, l))

    return output


def approximate_model(u, s, v):

    f = np.dot(np.dot(u, s), v.T)

    return f


def calculate_error(dimension,f, x):
    xtemp = transform_STZ(dimension, x, 1)
    script = "./shear_test2 qs"
    write_file(dimension, xtemp)
    Dpl = gfunSZT(script, 1)
    np.savetxt("Dplmodel.txt", Dpl)
    #np.savetxt("sample.txt", xtemp)

    psi, sigma, phi = project_2points(Dpl, f)

    error = gr_dist(psi[0], psi[1], 'Grassmann')

    return error

def project_2points(A, B):
    l = 2       # number of points  Gr(n, k)
    psi = []    # initialize the left singular values as a list
    sigma = []  # initialize the singular values as a list
    phi = []    # initialize the right singular values as a list

    # For each point perform svd with max_rank columns
    for i in range(l):
        if i == 0:
            u, s, v = svd(A, 0)
        else:
            u, s, v = svd(B, 0)

        psi.append(u)
        sigma.append(s)
        phi.append(v)

    return psi, sigma, phi

