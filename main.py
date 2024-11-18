import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import matplotlib
matplotlib.use('TkAgg')
from utils import *

# Example usage
N = 7  # Number of cars, you can increase this

# Initial conditions (generalize for N cars)
li = np.array([4, 4.4, 3.8, 5.2, 4.4, 3.8, 4], dtype=float)
xi = np.array([0, -8, -20, -40, -80, -100, -120], dtype=float)
vi = np.array([25, 27.8, 22.2, 19.4, 27.8, 22.2, 27.8], dtype=float)
ai = np.array([0, 2, 3, 2, 2, 3, 3])
desired_distance = np.array([0, 3, 4, 4, 3, 4, 3])

I = [[], [0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]

C = []
for i in I:
    C.append(len(i))

K = [3] * (N)
B = [5] * (N)
H = [1] * (N)
tau = [0.5] * (N)

simulation_time=10
time_step=0.005

num_of_iter = int(simulation_time / time_step)
time_array = np.arange(0, simulation_time, time_step)

cars = np.zeros((num_of_iter, N, 3))
cars_nonFilter = np.zeros((num_of_iter, N, 3))


xi, vi, ai = xi[:N], vi[:N], ai[:N]

A_matrix, B_matrix, C_matrix = get_matrices(N, K, B, H, I, C, tau)
U = get_U(N, K, I, tau, li, desired_distance)
print(U)
F = np.array([0]).reshape((1, 1))

del_xi, del_vi, del_ai = get_delta_vals(xi, vi, ai, N)
del_xi, del_vi, del_ai = del_xi[::-1], del_vi[::-1], del_ai[::-1]


X = np.concatenate((del_xi, del_vi, del_ai)).reshape((-1, 1))
X_current = X.copy()

for iter in range(num_of_iter):
    
    X_delta = A_matrix@X_current + B_matrix@U + C_matrix@F


    X_current = X_current + time_step * X_delta

    X_plot = X_current.copy()

    X_plot = X_plot.reshape(-1)
    new_del_xi = X_plot[: N]
    new_del_vi = X_plot[N: 2*N]
    new_del_ai = X_plot[2*N: 3*N]

    new_del_xi, new_del_vi, new_del_ai = new_del_xi[::-1], new_del_vi[::-1], new_del_ai[::-1]

    new_v0 = vi[0] + time_step * ai[0]
    new_x0 = xi[0] + time_step * vi[0]

    new_xi = new_del_xi + new_x0
    new_vi = new_del_vi + new_v0
    new_ai = new_del_ai + ai[0]

    xi = new_xi.copy()
    vi = new_vi.copy()
    ai = new_ai.copy()


    for i in range(N):
        cars[iter][i][0] = xi[i]
        cars[iter][i][1] = vi[i]
        cars[iter][i][2] = ai[i]


    print(cars[iter])
    print()

make_graph(N, time_array, cars, filter_flag=True, SNR=-1)

