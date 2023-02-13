import numpy as np
import random

# up-and-in, asset-or-nothing (at-Hit)
S = 95  # spot
X = 102  # exercise
T = 0.5
H = 100  # barrier
r = 0.1
b = 0.1
sigma = 0.2
K = 15  # return

N = 100000
N2 = int(252*T)
dt = T / N2

ret_lt = []
for _ in range(N):
    St = S
    S_lt = [St]
    ret = 0
    for ii in range(1, N2):
        St = S_lt[ii-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * random.gauss(0, 1))
        S_lt.append(St)
        if St >= H:
            ret = St*np.exp(-r*dt*ii)
            break
    ret_lt.append(ret)
# print(ret_lt)
print(np.mean(ret_lt))
