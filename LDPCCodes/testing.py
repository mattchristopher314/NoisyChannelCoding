import numpy as np


def LUP_factorisation(A):
    """Find P, L and U : PA = LU"""
    U = A.copy()
    shape_a = U.shape
    n = shape_a[0]
    L = np.eye(n)
    P = np.eye(n)
    for i in range(n):
        k = i
        comp = abs(U[i, i])
        for j in range(i, n):
            if abs(U[j, i]) > comp:
                k = j
                comp = abs(U[j, i])
        line_u = U[k, :].copy()
        U[k, :] = U[i, :]
        U[i, :] = line_u
        line_p = P[k, :].copy()
        P[k, :] = P[i, :]
        P[i, :] = line_p
        for j in range(i + 1, n):
            g = U[j, i] / U[i, i]
            L[j, i] = g
            U[j, :] -= g * U[i, :]
    return L, U, P


if __name__ == "__main__":
    A = np.array([[1., 3., 7.], [-2., 5., 4.], [2., 4., 90.]])
    L, U, P = LUP_factorisation(A)
    print(L @ U)
    print(P @ A)
