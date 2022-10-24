import copy
import sys
import random
import numpy as np
from PIL import Image, ImageOps


def GetParityBits(N, columnWeight, rowWeight):
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

    finalRowPermutation = np.random.permutation(H.shape[0])
    finalColumnPermutation = np.random.permutation(H.shape[1])

    H = H[:, finalColumnPermutation]
    H = H[finalRowPermutation, :]

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

    print(B.shape)

    s = np.array([1, 0, 0, 1, 1, 0])

    # P, L, U = PLU(A)
    # P = P % 2
    # L = L % 2
    # U = U % 2
    # xBar = P@B@s % 2
    # y = np.zeros(L.shape[1])

    # for i in range(len(y)):
    #     y[i] = xBar[i]-sum([L[i, j]*y[j] for j in range(i)])

    c = np.zeros(A.shape[1])
    # for i in range(len(c) - 1, -1, -1):
    #     c[i] = y[i]-sum([U[i, len(c)-1-j]*c[len(c)-1-j]
    #                     for j in range(len(c)-1-i)])

    cs = np.concatenate((c, s)) % 2
    invC = np.linalg.inv(A)@B@s

    return invC


def main():
    img = Image.open(sys.argv[1]).convert("L")
    img = ImageOps.exif_transpose(img)

    img.save("GreyscaleOriginal.png")
    print("Converted to greyscale.")

    pixels = img.load()

    blockSize = 6
    blocks = []

    currentBlock = []
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixelVal = [int(x)
                        for x in list(format(pixels[i, j], f"#010b")[2:])]
            for p in pixelVal:
                if len(currentBlock) >= blockSize:
                    blocks.append(currentBlock)
                    currentBlock = []
                currentBlock.append(p)
    if len(currentBlock) > 0:
        toAdd = blockSize - len(currentBlock)
        for i in range(toAdd):
            currentBlock.append(0)
        blocks.append(currentBlock)

    for block in blocks:
        parity = GetParityBits(16, 3, 4)
        for bit in parity:
            block.append(parity)

    f = 0.05

    for i in range(len(blocks)):
        for j in range(len(block)):
            if random.uniform(0, 1) < f:
                blocks[i][j] = 1 if blocks[i][j] == 0 else 0

    i = 0
    pixelCount = 0
    rawPixels = []
    for block in blocks:
        rawData = block[:blockSize]
        for b in rawData:
            rawPixels.append(b)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            pixel = "".join(
                [str(x) for x in rawPixels[(x*img.size[0]+y)*8:(x*img.size[0]+y)*8+8]])
            pixels[x, y] = int(pixel, 2)

    img.save(f"Noisy(f={f}).png")
    img.show()


if __name__ == "__main__":
    main()
