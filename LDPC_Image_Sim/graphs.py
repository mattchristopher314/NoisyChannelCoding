import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
y1 = np.array([0, 2/269, 235/4284, 268/9824, 1149/15967, 2122/22858])
y2 = np.array([0, 1/247, 0, 584/10923, 365/21594, 2518/23216])
y3 = np.array([0, 0, 1/5504, 56/12199, 397/15733, 2146/25519])
y = y1+y2+y3
y /= 3

plt.style.use("bmh")
plt.plot(x, y, 'x')
plt.errorbar(x, y, [(max(y1[i], y2[i], y3[i]) -
                     min(y1[i], y2[i], y3[i]))/2 for i in range(len(x))], ls="none", capsize=4, ecolor="#888")

# plt.xlim(0, 75)
plt.ylim(0, 0.1)

plt.title(
    r"Error Rates as a Function of BAWGN Variance $N$, $(3,4)$-regular Code, Block Size $16$")
plt.xlabel(r"$N$")
plt.ylabel(
    r"$\frac{\mathrm{Bit\ Errors\ in\ Decoded\ Image}}{\mathrm{Bit\ Errors\ in\ Noisy\ Image}}$")

plt.savefig("ErrorVsN.png", bbox_inches="tight", dpi=300)
plt.show()

x = np.array([4, 12, 16, 40, 64])
y1 = np.array([3900/7058, 601/9355, 497/9814, 145/22681, 6/29433])
y2 = np.array([3908/7265, 150/9350, 559/7033, 33/18777, 35/29065])
y3 = np.array([3978/7212, 752/7675, 61/12348, 48/23008, 67/28564])
y = y1+y2+y3
y /= 3

plt.style.use("bmh")
plt.plot(np.log(x), y, 'x')
plt.errorbar(np.log(x), y, [(max(y1[i], y2[i], y3[i]) -
                             min(y1[i], y2[i], y3[i]))/2 for i in range(len(x))], ls="none", capsize=4, ecolor="#888")

# plt.xlim(0, 75)
plt.ylim(0, 0.6)

plt.title(
    r"Error Rates as a Function of Block Size $N$, $(3,4)$-regular Code, Variance $0.25$")
plt.xlabel(r"$\log N$")
plt.ylabel(
    r"$\frac{\mathrm{Bit\ Errors\ in\ Decoded\ Image}}{\mathrm{Bit\ Errors\ in\ Noisy\ Image}}$")

plt.savefig("ErrorVsBlock.png", bbox_inches="tight", dpi=300)
plt.show()
