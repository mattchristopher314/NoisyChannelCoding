import numpy as np
import math


def parityCheck(n, rowDensity, columnDensity):
    R = int(n * columnDensity / rowDensity)

    HFirstSubmatrix = np.zeros((int(R/columnDensity), n), dtype=int)

    for i in range(int(R/columnDensity)):
        for j in range(rowDensity):
            HFirstSubmatrix[i, rowDensity*i+j] = 1

    secondSubmatrixPermutation = np.random.permutation(
        HFirstSubmatrix.shape[1])
    thirdSubmatrixPermutation = np.random.permutation(
        HFirstSubmatrix.shape[1])

    H = np.vstack((HFirstSubmatrix,
                   HFirstSubmatrix[:, secondSubmatrixPermutation], HFirstSubmatrix[:, thirdSubmatrixPermutation]))

    return H


def generator(H):
    Href = gaussjordan(H)

    pivots = []

    rank = 0
    for i, r in enumerate(Href):
        if sum(r) > 0:
            rank += 1
        leadOneIndex = 0
        leadIndexSet = False
        for j in range(len(r)):
            if r[j] == 1 and not (leadIndexSet):
                leadOneIndex = j
                leadIndexSet = True
                pivots.append(leadOneIndex)
    nonPivots = [x for x in range(len(Href[0])) if not (x in pivots)]

    G = np.zeros(shape=(H.shape[1]-rank, H.shape[1]))

    nonZeroReductionRows = [i for i in range(
        Href.shape[0]) if sum(Href[i]) > 0]
    for i in range(len(nonPivots)):
        G[i, nonPivots[i]] = 1
        for j in range(len(pivots)):
            G[i, pivots[j]] = Href[nonZeroReductionRows[j], nonPivots[i]]

    return G


def gausselimination(A, b):
    # Via https://github.com/hichamjanati/pyldpc/blob/a821ccd1eb3a13b8a0f66ebba8d9923ce2f528ef/pyldpc/utils.py#L161
    """Solve linear system in Z/2Z via Gauss Gauss elimination."""
    A = A.copy()
    b = b.copy()
    n, k = A.shape

    for j in range(min(k, n)):
        listedepivots = [i for i in range(j, n) if A[i, j]]
        if len(listedepivots):
            pivot = np.min(listedepivots)
        else:
            continue
        if pivot != j:
            aux = (A[j, :]).copy()
            A[j, :] = A[pivot, :]
            A[pivot, :] = aux

            aux = b[j].copy()
            b[j] = b[pivot]
            b[pivot] = aux

        for i in range(j+1, n):
            if A[i, j]:
                A[i, :] = abs(A[i, :]-A[j, :])
                b[i] = abs(b[i]-b[j])

    return A, b


def extract_message_bits(G, encodedVector):
    # Via https://github.com/hichamjanati/pyldpc/blob/a821ccd1eb3a13b8a0f66ebba8d9923ce2f528ef/pyldpc/decoder.py#L186
    n, k = G.shape

    rtG, rx = gausselimination(G, encodedVector)

    message = np.zeros(k).astype(int)

    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= (rtG[i, list(range(i+1, k))] @
                       message[list(range(i+1, k))]) % 2

    return abs(message)


def gaussjordan(X, change=0):
    # Via https://github.com/hichamjanati/pyldpc/blob/a821ccd1eb3a13b8a0f66ebba8d9923ce2f528ef/pyldpc/utils.py#L38
    """Compute the binary row reduced echelon form of X.
    Parameters
    ----------
    X: array (m, n)
    change : boolean (default, False). If True returns the inverse transform
    Returns
    -------
    if `change` == 'True':
        A: array (m, n). row reduced form of X.
        P: tranformations applied to the identity
    else:
        A: array (m, n). row reduced form of X.
    """
    A = np.copy(X)
    m, n = A.shape

    if change:
        P = np.identity(m).astype(int)

    pivot_old = -1
    for j in range(n):
        filtre_down = A[pivot_old+1: m, j]
        pivot = np.argmax(filtre_down)+pivot_old+1

        if A[pivot, j]:
            pivot_old += 1
            if pivot_old != pivot:
                aux = np.copy(A[pivot, :])
                A[pivot, :] = A[pivot_old, :]
                A[pivot_old, :] = aux
                if change:
                    aux = np.copy(P[pivot, :])
                    P[pivot, :] = P[pivot_old, :]
                    P[pivot_old, :] = aux

            for i in range(m):
                if i != pivot_old and A[i, j]:
                    if change:
                        P[i, :] = abs(P[i, :]-P[pivot_old, :])
                    A[i, :] = abs(A[i, :]-A[pivot_old, :])

        if pivot_old == m-1:
            break

    if change:
        return A, P
    return A


def dataToCodeword(G, inputVector):
    K, N = G.shape
    data = ((G.T)@inputVector) % 2
    codeWord = (-1)**data

    return codeWord


def encode_image(G, binaryMatrix):
    K, N = G.shape

    flatInput = binaryMatrix.flatten()
    bitsToProcess = len(flatInput)

    blocks = bitsToProcess // K
    leftOver = blocks % K
    if (leftOver) > 0:
        blocks += 1

    # Pad the image with zeros.
    padded = np.zeros(K * blocks)
    padded[:bitsToProcess] = flatInput

    # Rearrange the binary matrix into its blocks
    blockedImage = padded.reshape(K, blocks)
    encoded = dataToCodeword(G, blockedImage)

    return encoded


def decode_image(G, H, noiseVariance, binaryMatrix, img_shape):
    K, N = H.shape

    extracted = []

    MAX_ITER = 20

    appLLRForConvergence = np.zeros(
        shape=(len(binaryMatrix[:, 0]), MAX_ITER+1))

    thing = np.random.randint(0, len(binaryMatrix.T))
    for r, vector in enumerate(binaryMatrix.T):
        print(r+1, len(binaryMatrix.T))
        # Initialisation phase - prior probability vector.
        v = -2*vector/noiseVariance
        # Initialisation phase - messages.
        MvcLLR = np.zeros(shape=(N, K))
        McvLLR = np.zeros(shape=(K, N))

        for i in range(MAX_ITER):
            # Data-check Message-passing and Update Phase
            # Update LLR messages for each data node over all check adjacencies.

            for n in range(N):
                for c in range(K):
                    if H[c][n] == 1:
                        MvcLLR[n][c] = v[n] + sum([McvLLR[j][n]
                                                  for j in range(K) if H[j][n] == 1 and j != c])

            # Stopping Criterion Check and Iteration Decision Phase
            # Check satisfaction of parity constraints.
            checkPassed = True
            for j in range(K):
                # print("Satisfaction", j+1, np.prod([MvcLLR[n][j]
                #       for n in range(N) if H[j][n] == 1]))
                if np.prod([MvcLLR[n][j] for n in range(N) if H[j][n] == 1]) <= 0:
                    checkPassed = False
            if checkPassed:
                appLLRForConvergence[:, 1] = v
                break

            # Check-data Message-passing and Update Phase
            # Update LLR messages for each check node over all message adjacencies.
            for j in range(K):
                for l in range(N):
                    if H[j][l] == 1:
                        McvLLR[j][l] = 2 * \
                            math.atanh(
                                np.clip(np.prod([math.tanh(MvcLLR[n][j]/2) for n in range(N) if H[j][n] == 1 and n != l]), -0.9999, 0.9999))

            if r == thing:
                vPrime = np.zeros(N)
                for n in range(len(vPrime)):
                    vPrime[n] = v[n] + sum([McvLLR[c][n]
                                            for c in range(K) if H[c][n] == 1])
                    appLLRForConvergence[n][i+1] = vPrime[n]

        # Output Phase - a posteriori.
        vPrime = np.zeros(N)
        for n in range(len(vPrime)):
            vPrime[n] = v[n] + sum([McvLLR[c][n]
                                    for c in range(K) if H[c][n] == 1])

        # Output Phase - message bit extraction.
        extracted.append(extract_message_bits(G.T, (vPrime > 0).astype(int)))

    for i in range(len(binaryMatrix[:, 0])):
        curLatest = 0
        for j in range(MAX_ITER):
            if appLLRForConvergence[i][j+1] != 0:
                curLatest = appLLRForConvergence[i][j+1]
            else:
                appLLRForConvergence[i][j+1] = curLatest

    return np.array(extracted).T, appLLRForConvergence


def initialise_ldpc(n, rowDensity, columnDensity):
    print("Generating parity-check matrix -", end="")
    H = parityCheck(n, rowDensity, columnDensity)
    # H = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], [
    #              0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]])
    print(" done.")
    print("Generating generator matrix -", end="")
    G = generator(H)
    print(" done.")

    return H, G
