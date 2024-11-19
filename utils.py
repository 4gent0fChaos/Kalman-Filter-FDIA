import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

def compare_with_threshold(matrix1, matrix2, threshold):
    # Compute the absolute difference between the matrices
    difference = np.abs(matrix1 - matrix2)
    
    # Check if the difference exceeds the threshold
    result = np.any(difference > threshold)
    return result

def add_noise_based_on_snr(signal, snr_db):
    # Calculate the signal power
    signal_power = np.mean(signal**2)
    
    # Convert SNR from decibels to linear scale
    snr_linear = 10 ** (snr_db / 10.0)
    
    # Calculate the noise power
    noise_power = signal_power / snr_linear
    
    # Generate Gaussian noise with the calculated noise power
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    # Add the noise to the original signal
    noisy_signal = signal + noise
    
    return noisy_signal

# Kalman filter function
def kalman_filter(X, P, A, B, U, H, Q, R, Z):

    epsilon =  1e-5
    lambda_val = 0.005

    # Prediction step
    X_pred = np.dot(A, X) + np.dot(B, U)
    P_pred = np.dot(A, np.dot(P, A.T)) + Q

    Y = Z - np.dot(H, X_pred)
    
    # Update step
    S = np.dot(H, np.dot(P_pred, H.T)) + R
    K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))

    X_new = X_pred + np.dot(K, Y)

    I = np.eye(P.shape[0])  # Identity matrix of size P
    P_new = np.dot(I - np.dot(K, H), P_pred)

    return X_new, P_new

def kalman_filter_pwls(X, P, A, B, U, H, Q, R, Z):
    epsilon =  1e-10
    lambda_val = 0.005

    # Prediction step
    X_pred = np.dot(A, X) + np.dot(B, U)
    P_pred = np.dot(A, np.dot(P, A.T)) + Q

    Y = Z - np.dot(H, X_pred)
    
    # Update step
    S = np.dot(H, np.dot(P_pred, H.T)) + R
    K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))

    z_bar = np.vstack((Y, X_pred))

    H_bar = np.vstack((H, H))

    # PWLS Iterative Process
    w_est = np.vstack((np.zeros((len(Y), 1)), np.ones((len(Y), 1))))  # Start with all weights as 1
    eps = 1
    X_plus = X_pred.copy()  # Initialize with predicted state

    Sk = block_diag(R, P_pred)

    X_new = X_plus
    residual_new = z_bar - H_bar @ X_new

    while eps > epsilon and np.linalg.norm(residual_new) > 10:
        Z_adj = np.dot(np.diag(w_est.flatten()), z_bar)
        H_adj = np.dot(np.diag(w_est.flatten()), H_bar)

        X_old = X_new

        M = H_adj.T @ np.linalg.inv(Sk) @ H_adj
        X_new = np.linalg.inv(M) @ (H_adj.T @ np.linalg.inv(Sk) @ Z_adj)

        residual_new = z_bar - H_bar @ X_new
        
        X_plus = X_new.copy()

        w_est = np.maximum(np.minimum(np.sqrt(0.5 * lambda_val) / np.abs(residual_new), 1), 0)
        eps = np.linalg.norm(X_old - X_new, np.inf)

    w_est[w_est < 0.5] = 0

    Z_adj = np.dot(np.diag(w_est.flatten()), z_bar)
    H_adj = np.dot(np.diag(w_est.flatten()), H_bar)
    M = H_adj.T @ np.linalg.inv(Sk) @ H_adj
    X_new = np.linalg.inv(M) @ (H_adj.T @ np.linalg.inv(Sk) @ Z_adj)
    P_new = np.dot((np.eye(len(K)) - np.dot(K, H)), P_pred)

    res = np.linalg.norm(residual_new)
    
    return X_new, P_new, res



# Function to dynamically create matrices based on N cars
def get_matrices(N, K, B, H, I, C, tau):
    Z_NN = np.zeros((N, N))
    I_NN = np.eye(N)

    Z_N = np.zeros((N, 1))
    Z_n = np.zeros((N-1, 1))

    Xi = Z_NN.copy()
    Omega = Z_NN.copy()
    Lambda = Z_NN.copy()
    
    for i in range(N):
        for j in range(N):
            if (i == j):
                Xi[i][j] = -1 * C[N-1-i] * K[N-1-i] / tau[N-1-i]
                Omega[i][j] = -1 * C[N-1-i] * B[N-1-i] / tau[N-1-i]
                Lambda[i][j] = -1 * (1 + C[N-1-i] * H[N-1-i]) / tau[N-1-i]
            else:
                S = get_S(N-1-i, N-1-j, I)
                Xi[i][j] = K[N-1-i] / tau[N-1-i] * S
                Omega[i][j] = B[N-1-i] / tau[N-1-i] * S
                Lambda[i][j] = H[N-1-i] / tau[N-1-i] * S

            

    C_matrix = np.zeros((3*N, 1))
    C_matrix[-1, 0] = 1 
    
    A_matrix = np.block([
        [Z_NN, I_NN, Z_NN],
        [Z_NN, Z_NN, I_NN],
        [Xi, Omega, Lambda]
    ])
    B_matrix = np.block([
        [Z_NN],
        [Z_NN],
        [I_NN]
    ])
    return A_matrix, B_matrix, C_matrix


# Function to calculate control inputs based on the desired distances
def get_U(N, K, I, tau, li, desired_distance):
    U = np.zeros((N, 1))
    for k in range(0, N-1):
        sum_distances = 0
        mat_K = N-k-1
        for j in I[mat_K]:  # Car k follows car 0 and the car directly in front
            sum_distances += get_desired_distance(mat_K, j, li, desired_distance)
    
        U[k] = K[k] / tau[k] * sum_distances
    return U

def get_S(i, j, I):
    if j in I[i]:
        return 1
    return 0

def sgn(i, j):
    return 1 if i > j else -1

# Function to calculate the desired distance between cars
def get_desired_distance(i, j, li, desired_distance):
    d = 0
    for k in range(min([i, j]), max([i, j])):
        d += li[k] + desired_distance[k+1]
    return -1 * sgn(i, j) * d

def get_beta0(maintain_vel, current_acc, tau0):
    beta0 = tau0 * maintain_vel + (tau0-1) * current_acc
    return beta0

def get_delta_vals(xi, vi, ai, N):
    # Compute deltas (differences between cars and the leader, car 0)
    xi = ((xi - xi[0]))
    vi = ((vi - vi[0]))
    ai = ((ai - ai[0]))
    return xi, vi, ai

def get_val_from_delta(X_plot, time_step, N, xi, vi, ai):
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
    return xi, vi, ai

def make_graph(N, time_array, cars, filter_flag, SNR):
    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 0], label=f"Car {car} (Follower)", linestyle='-')

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    if filter_flag:
        plt.title(f"Position of Cars in Platoon over Time (After filters) ({SNR}db)")
    else:
        plt.title(f"Position of Cars in Platoon over Time (Without filters) ({SNR}db)")

    plt.legend()
    plt.grid(True)
    if filter_flag:
        plt.savefig("platoon_positions.png")
    else:
        plt.savefig("platoon_positions_filterless.png")


    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 1], label=f"Car {car} (Follower)", linestyle='-')

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    if filter_flag:
        plt.title(f"Velocity of Cars in Platoon over Time (After filters) ({SNR}db)")
    else:
        plt.title(f"Velocity of Cars in Platoon over Time (Without filters) ({SNR}db)")

    plt.legend()
    plt.grid(True)
    if filter_flag:
        plt.savefig("platoon_velocities.png")
    else:
        plt.savefig("platoon_velocities_filterless.png")
        

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 2], label=f"Car {car} (Follower)", linestyle='-')

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s2)")
    if filter_flag:
        plt.title(f"Acceleration of Cars in Platoon over Time (After filters) ({SNR}db)")
    else:
        plt.title(f"Acceleration of Cars in Platoon over Time (Wihtout filters) ({SNR}db)")

    plt.legend()
    plt.grid(True)
    if filter_flag:
        plt.savefig("platoon_accelerations.png")
    else:
        plt.savefig("platoon_accelerations_filterless.png")

class UnknownInputKalmanFilter:
    def __init__(self, A, B, C, Q, R, P0):
        """
        Initializes the Kalman Filter.
        :param A: State transition matrix
        :param B: Control input matrix
        :param C: Unknown input matrix
        :param Q: Process noise covariance
        :param R: Measurement noise covariance
        :param P0: Initial error covariance matrix
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.P = P0

    def predict(self, X, U):
        """
        Predicts the next state and error covariance.
        :param X: Current state estimate
        :param U: Control input
        :return: Predicted state estimate
        """
        self.X_pred = self.A @ X + self.B @ U
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.X_pred

    def update(self, Z, H):
        """
        Updates the state and error covariance based on measurement.
        :param Z: Measurement vector
        :param H: Observation matrix
        :return: Updated state estimate
        """
        # Calculate Kalman Gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Estimate unknown input F (if C is invertible)
        residual = Z - H @ self.X_pred
        if self.C.shape[0] == self.C.shape[1] and np.linalg.det(self.C) != 0:
            F_est = np.linalg.inv(self.C) @ residual
        else:
            F_est = np.zeros((self.C.shape[1], 1))  # Use zero if estimation is not possible
        
        # Update state estimate
        self.X_pred += K @ residual
        self.P = (np.eye(len(self.P)) - K @ H) @ self.P

        return self.X_pred, F_est