import numpy as np
from scipy.linalg import block_diag

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


    # z_bar = np.vstack((residual, X_pred))

    # H_bar = np.vstack((H, H))

    # # PWLS Iterative Process
    # w_est = np.vstack((np.zeros((len(residual), 1)), np.ones((len(residual), 1))))  # Start with all weights as 1
    # eps = 1
    # X_plus = X_pred.copy()  # Initialize with predicted state

    # Sk = block_diag(R, P_pred)

    # X_new = X_plus

    
    # while eps > epsilon:
    #     Z_adj = np.dot(np.diag(w_est.flatten()), z_bar)
    #     H_adj = np.dot(np.diag(w_est.flatten()), H_bar)

    #     X_old = X_new

    #     M = H_adj.T @ np.linalg.inv(Sk) @ H_adj
    #     X_new = np.linalg.inv(M) @ (H_adj.T @ np.linalg.inv(Sk) @ Z_adj)

    #     residual_new = z_bar - H_bar @ X_new
        
    #     X_plus = X_new.copy()

    #     w_est = np.maximum(np.minimum(np.sqrt(0.5 * lambda_val) / np.abs(residual_new), 1), 0)
    #     eps = np.linalg.norm(X_old - X_new, np.inf)



    # Z_adj = np.dot(np.diag(w_est.flatten()), z_bar)
    # H_adj = np.dot(np.diag(w_est.flatten()), H_bar)
    # M = H_adj.T @ np.linalg.inv(Sk) @ H_adj
    # X_new = np.linalg.inv(M) @ (H_adj.T @ np.linalg.inv(Sk) @ Z_adj)
    # P_new = np.dot((np.eye(len(K)) - np.dot(K, H)), P_pred)

    return X_new, P_new


# Function to dynamically create matrices based on N cars
def get_matrices(N, K, B, H, I, C, tau):
    Z_n_n = np.zeros((N-1, N-1))
    I_n_n = np.eye(N-1)

    Xi = Z_n_n.copy()
    Omega = Z_n_n.copy()
    Lambda = Z_n_n.copy()
    
    for i in range(N-1):
        for j in range(N-1):
            if (i == j):
                Xi[i][j] = -1 * C[N-1-i] * K[N-1-i] / tau[N-1-i]
                Omega[i][j] = -1 * C[N-1-i] * B[N-1-i] / tau[N-1-i]
                Lambda[i][j] = -1 * (1 + C[N-1-i] * H[N-1-i]) / tau[N-1-i]
            else:
                S = get_S(N-1-i, N-1-j, I)
                Xi[i][j] = K[N-1-i] / tau[N-1-i] * S
                Omega[i][j] = B[N-1-i] / tau[N-1-i] * S
                Lambda[i][j] = H[N-1-i] / tau[N-1-i] * S
    
    A_matrix = np.block([
        [Z_n_n, I_n_n, Z_n_n],
        [Z_n_n, Z_n_n, I_n_n],
        [Xi, Omega, Lambda]
    ])
    B_matrix = np.block([
        [Z_n_n],
        [Z_n_n],
        [I_n_n]
    ])
    
    return A_matrix, B_matrix

# Function to calculate control inputs based on the desired distances
def get_U(N, K, I, tau, li, desired_distance):
    U = np.zeros((N-1, 1))
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