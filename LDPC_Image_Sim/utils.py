import numpy as np


def greyscale_to_binary(img):
    h, w = img.shape

    matrix = np.zeros(shape=(h, w, 8), dtype=int)

    for y in range(h):
        for x in range(w):
            matrix[y, x, :] = [int(x) for x in list(
                format(img[y, x], f"#010b")[2:])]

    return matrix


def binary_to_greyscale(matrix):
    img_x, img_y, img_z = matrix.shape
    img = np.zeros(shape=(img_x, img_y), dtype=int)

    for x in range(img_x):
        for y in range(img_y):
            img[x, y] = int("".join(str(x) for x in matrix[x, y, :]), 2)

    return img
