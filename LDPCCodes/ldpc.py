import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import scipy as sp
import copy
import sys
import random
from PIL import Image, ImageOps


def ldpc_sim(N, columnWeight, rowWeight):
    R = int(N * columnWeight / rowWeight)

    HFirstSubmatrix = np.zeros((int(R/columnWeight), N))

    for i in range(int(R/columnWeight)):
        for j in range(rowWeight):
            HFirstSubmatrix[i, rowWeight*i+j] = 1

    secondSubmatrixPermutation = np.random.permutation(
        HFirstSubmatrix.shape[1])
    thirdSubmatrixPermutation = np.random.permutation(
        HFirstSubmatrix.shape[1])

    H = np.vstack((HFirstSubmatrix,
                   HFirstSubmatrix[:, secondSubmatrixPermutation], HFirstSubmatrix[:, thirdSubmatrixPermutation]))

    # finalRowPermutation = np.random.permutation(H.shape[0])
    # finalColumnPermutation = np.random.permutation(H.shape[1])

    # H = H[:, finalColumnPermutation]
    # H = H[finalRowPermutation, :]

    H = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], [
                 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]])
    print(H)

    G = nx.Graph()

    G.add_nodes_from([f"C{i}" for i in range(R)], bipartite=0)
    G.add_nodes_from(range(N), bipartite=1)

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if H[i, j] == 1:
                G.add_edge(f"C{i}", j)

    pos = nx.drawing.layout.bipartite_layout(
        G, [f"C{i}" for i in range(R)], align='horizontal')

    checkPos = [pos[f"C{i}"] for i in range(R)]
    checkPos.sort(key=lambda x: x[0])

    for i in range(R):
        pos[f"C{i}"] = checkPos[i]

    nx.draw_networkx_edges(G, pos,
                           width=0.4)

    for i, node in enumerate(G.nodes()):
        G.nodes[node]['shape'] = 's' if i < R else 'o'

    for shape in ['o', 's']:
        node_list = [node for node in G.nodes() if G.nodes[node]
                     ['shape'] == shape]

        nx.draw_networkx_nodes(G, pos,
                               nodelist=node_list,
                               node_size=150 if shape == 's' else 175,
                               node_color='black',
                               node_shape=shape)
        nx.draw_networkx_labels(
            G, pos=dict([(key, value) for key, value in [(n, [pos[n][0]+0.003, pos[n][1]+0.1*(1 if not ("C" in str(n)) else -1.25)]) for n in G]]), labels={n: int(
                str(n).replace('C', ''))+1 for n in G}, font_color='black', font_size=13)

    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.xlabel("Check (Constraint) Nodes", fontsize=13)
    plt.title("Message (Data) Nodes", fontsize=13)

    plt.savefig("TannerGraph.png", dpi=300, bbox_inches='tight', pad_inches=0)

    Hcopy = []

    while (True):
        Hcopy = H[:, :]
        rowsToDrop = np.random.choice(
            H.shape[0], R-np.linalg.matrix_rank(H), replace=False)
        Hcopy = np.delete(Hcopy, rowsToDrop, 0)
        if (np.linalg.matrix_rank(Hcopy) == np.linalg.matrix_rank(H)):
            break

    K = N-Hcopy.shape[0]

    aux = []
    while (True):
        aux = Hcopy[:, np.random.permutation(Hcopy.shape[1])]
        if (np.linalg.det(aux[:, :N-K]) % 2 == 1):
            break

    Hcopy = aux[:]
    A = Hcopy[:, :N-K]
    B = Hcopy[:, N-K:]

    s = np.array([1, 0, 0, 1, 1, 0])

    P, L, U = PLU(A)
    # print("Test 1:", (P.T@L@U-A) % 2)
    P = P % 2
    L = L % 2
    U = U % 2
    # print("PMATRIX")
    # print("UMATRIX")
    # print(U)
    xBar = P@B@s % 2
    y = np.zeros(L.shape[1])

    for i in range(len(y)):
        y[i] = xBar[i]-sum([L[i, j]*y[j] for j in range(i)])

    # print("Test 2:", (L@y-xBar) % 2)

    c = np.zeros(U.shape[1])
    for i in range(len(c) - 1, -1, -1):
        c[i] = y[i]-sum([U[i, len(c)-1-j]*c[len(c)-1-j]
                         for j in range(len(c)-1-i)])

    # print("Test 3: ", (U@c-y) % 2)

    # print("Final encoding check:")

    cs = np.concatenate((c, s)) % 2
    invC = np.linalg.inv(A)@B@s
    # print((Hcopy@np.concatenate((invC, s))) % 2)
    # print((Hcopy@cs) % 2)


def PermutationMatrix(A):
    n = A.shape[0]
    P = np.eye(n, dtype=np.double)

    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(A[i][j]))
        if j != row:
            P[[j, row]] = P[[row, j]]
    return P


def PLU(A):
    n = A.shape[0]

    U = A.copy()
    P = PermutationMatrix(U)
    U = P
    L = np.eye(n, dtype=np.double)

    for i in range(n):
        factor = U[i+1:, i]
        L[i+1:, i] = factor % 2
        U[i+1:] -= (factor[:, np.newaxis] * U[i]) % 2

    return P, L, U@P


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    ldpc_sim(16, 3, 4)
    # img = Image.open(sys.argv[1]).convert("L")
    # img = ImageOps.exif_transpose(img)

    # img.save("GreyscaleOriginal.png")
    # print("Converted to greyscale.")

    # pixels = img.load()

    # blocks = []
    # decoderBlocks = []

    # data = []

    # for i in range(img.size[0]):
    #     for j in range(img.size[1]):
    #         pixelVal = [int(x)
    #                     for x in list(format(pixels[i, j], f"#010b")[2:])]
    #         for x in pixelVal:
    #             data.append(pixelVal)

    # print(data)

    # correct = copy.deepcopy(blocks)


if __name__ == "__main__":
    main()
