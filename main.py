import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import matplotlib
matplotlib.use('TkAgg')
from utils import *
from utilsGraph import *

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

beta_0 = 0

simulation_time=10
time_step=0.005

num_of_iter = int(simulation_time / time_step)
time_array = np.arange(0, simulation_time, time_step)

cars = np.zeros((num_of_iter, N, 3))
cars_noisy = np.zeros((num_of_iter, N, 3))

xi, vi, ai = xi[:N][::-1], vi[:N][::-1], ai[:N][::-1]

A_matrix, B_matrix, C_matrix = get_matrices(N, K, B, H, I, C, tau)
U = get_U(N, K, I, tau, li, desired_distance)

X = np.concatenate((xi, vi, ai)).reshape((-1, 1))
X_current = X.copy()
X_noisy = X.copy()

trueF = []
estF= []
resList = []
beta_agg = []

SNR_measurement = 30  # Signal-to-Noise Ratio for measurements
SNR_process = 100

# Kalman Filter Variables
H_kalman = np.eye(3 * (N))
Q_kalman = noise_to_kalman_covariance(SNR_process, matrix_type='R', state_dim=(3*N))
R_kalman = noise_to_kalman_covariance(SNR_measurement, matrix_type='R', state_dim=(3*N)) # Measurement noise covariance
P = np.eye(3 * (N)) * 0.5  # Initial state covariance



uikf = ModifiedUIKF_pwls(A_matrix, B_matrix, C_matrix, H_kalman, Q_kalman, R_kalman, P)

for iter in range(num_of_iter):

    current_time = iter * time_step

    if current_time > 4 and current_time < 6 and beta_0 < 10:
        beta_0 += 1

    if current_time > 6 and beta_0 > 0:
        beta_0 -= 1

    beta_agg.append(beta_0)


    F = np.array([beta_0]).reshape((1, 1))
    trueF.append(F[0][0])

    Z_measured = A_matrix@X_current + B_matrix@U + C_matrix@F
    Z_noisy = add_noise_based_on_snr(Z_measured, SNR_measurement)

    X_delta_noisy = A_matrix@X_noisy + B_matrix@U + C_matrix@F
    X_delta_noisy = add_noise_based_on_snr(X_delta_noisy, SNR_measurement)

    if current_time > 5 and current_time < 6:
        Z_noisy[6] += 50    # Car 0 position
        X_delta_noisy[6] += 50

    if current_time > 6.5 and current_time < 7.5:
        Z_noisy[-2] += 25    # Car 1 acceleration
        X_delta_noisy[-2] += 25

    if current_time > 8 and current_time < 9:
        Z_noisy[11] += 40    # Car 2 velocity
        X_delta_noisy[11] += 40


    X_pred = uikf.predict(X_current, U)
    X_filtered, F_est, res = uikf.update(Z_noisy)

    # res = np.linalg.norm(X_filtered - Z_noisy)

    estF.append(F_est[0][0])
    resList.append(res)

    X_current = X_current + time_step * X_filtered

    X_noisy = X_noisy + time_step * X_delta_noisy

    # For Plotting    
    X_plot = X_current.copy()
    xi = X_plot[: N].reshape((-1))
    vi = X_plot[N: 2*N].reshape((-1))
    ai = X_plot[2*N: 3*N].reshape((-1))

    X_plot_noisy = X_noisy.copy()
    xi_noisy = X_plot_noisy[:N].reshape((-1))
    vi_noisy = X_plot_noisy[N: 2*N].reshape((-1))
    ai_noisy = X_plot_noisy[2*N: 3*N].reshape((-1))


    for i in range(N):
        cars[iter][i][0] = xi[i]
        cars[iter][i][1] = vi[i]
        cars[iter][i][2] = ai[i]

        cars_noisy[iter][i][0] = xi_noisy[i]
        cars_noisy[iter][i][1] = vi_noisy[i]
        cars_noisy[iter][i][2] = ai_noisy[i]
        



make_graph(N, time_array, cars, SNR=10)
make_graph_filterless(N, time_array, cars_noisy, SNR=10)

make_true_est_graph(trueF, estF, time_array)

make_graph_res(resList, time_array)


plt.figure(figsize=(10, 6))
for car in range(N):
    plt.plot(time_array, beta_agg)

plt.xlabel("Time (s)")
plt.ylabel("Position (m)")

plt.legend()
plt.grid(True)
plt.savefig("fig/beta.png")

