import numpy as np
from scipy import integrate
from scipy.stats import norm
import matplotlib.pyplot as plt
import math


def phi(y, N):
    res = 1/math.sqrt(8*math.pi*N) * \
        (math.exp(-(y-1)**2/(2*N))+math.exp(-(y+1)**2/(2*N)))
    return res


def f(y, N):
    return phi(y, N)*math.log2(phi(y, N))


N = np.arange(0.1, 1000, 0.01)
approximationRadius = 10


def F(n):
    res = np.zeros_like(n)
    for i, val in enumerate(n):
        y, err = integrate.quad(f, -approximationRadius,
                                approximationRadius, args=val)
        res[i] = y
    return res


def G(n):
    n = 1-norm.cdf(n)
    return 1-(n*math.log2(1/n)+(1-n)*math.log2(1/(1-n)))


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.figsize": (4.5, 2)
})

plt.style.use("bmh")

plt.xlabel(r"$\frac{1}{N}$")
plt.ylabel("Capacity")

plt.xlim(0.1, 10)
plt.ylim(0, 1)

plt.grid(True)

plt.plot(1/N, -F(N)-[1/2*math.log2(2*math.pi*math.e*n)
         for n in N], label="BAWGN")
print(-F(np.array([0.51188])) -
      [1/2*math.log2(2*math.pi*math.e*np.array([0.51188]))])

plt.savefig("NumericalCapacity.png", dpi=300,
            bbox_inches='tight', pad_inches=0)


plt.show()
