import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# Kalman filter function
def kalman_filter(X, P, A, B, U, H, Q, R, Z):

    epsilon =  1e-5
    lambda_val = 0.005

    # Prediction step
    X_pred = np.dot(A, X) + np.dot(B, U)
    P_pred = np.dot(A, np.dot(P, A.T)) + Q
    
    # Update step
    S = np.dot(H, (np.dot(P_pred, H.T) + R))
    K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))

    residual = Z - np.dot(H, X_pred)

    z_bar = np.vstack((residual, X_pred))

    H_bar = np.vstack((H, H))

    # PWLS Iterative Process
    w_est = np.vstack((np.zeros((len(residual), 1)), np.ones((len(residual), 1))))  # Start with all weights as 1
    eps = 1
    X_plus = X_pred.copy()  # Initialize with predicted state

    Sk = block_diag(R, P_pred)

    X_new = X_plus

    
    while eps > epsilon:
        Z_adj = np.dot(np.diag(w_est.flatten()), z_bar)
        H_adj = np.dot(np.diag(w_est.flatten()), H_bar)

        X_old = X_new

        M = H_adj.T @ np.linalg.inv(Sk) @ H_adj
        X_new = np.linalg.inv(M) @ (H_adj.T @ np.linalg.inv(Sk) @ Z_adj)

        residual_new = z_bar - H_bar @ X_new
        
        X_plus = X_new.copy()

        w_est = np.maximum(np.minimum(np.sqrt(0.5 * lambda_val) / np.abs(residual_new), 1), 0)
        eps = np.linalg.norm(X_old - X_new, np.inf)



    Z_adj = np.dot(np.diag(w_est.flatten()), z_bar)
    H_adj = np.dot(np.diag(w_est.flatten()), H_bar)
    X_old = X_new
    M = H_adj.T @ np.linalg.inv(Sk) @ H_adj
    X_new = np.linalg.inv(M) @ (H_adj.T @ np.linalg.inv(Sk) @ Z_adj)
    P_new = np.dot((np.eye(len(K)) - np.dot(K, H)), P_pred)

    return X_new, P_new


# Function to dynamically create matrices based on N cars
def get_matrices(N, K, B, H, I):
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
def get_U(N, K, I):
    U = np.zeros((N-1, 1))
    for k in range(0, N-1):
        sum_distances = 0
        mat_K = N-k-1
        for j in I[mat_K]:  # Car k follows car 0 and the car directly in front
            sum_distances += get_desired_distance(mat_K, j)
    
        U[k] = K[k] / tau[k] * sum_distances
    return U

def get_S(i, j, I):
    if j in I[i]:
        return 1
    return 0

def sgn(i, j):
    return 1 if i > j else -1

# Function to calculate the desired distance between cars
def get_desired_distance(i, j):
    d = 0
    for k in range(min([i, j]), max([i, j])):
        d += li[k] + desired_distance[k+1]
    return -1 * sgn(i, j) * d

# Generalized function to handle N cars
def simulate_platoon(N, I, li, xi, vi, ai, desired_distance, C, K, B, H, tau, simulation_time=10, time_step=0.005):
    
    # Ensure lists have correct size based on platoon size N
    C, K, B, H, tau = [C[:N+1], K[:N+1], B[:N+1], H[:N+1], tau[:N+1]]


    # Compute deltas (differences between cars and the leader, car 0)
    del_xi = (xi[1:] - xi[0])[::-1].reshape(1, -1)
    del_vi = (vi[1:] - vi[0])[::-1].reshape(1, -1)
    del_ai = (ai[1:] - ai[0])[::-1].reshape(1, -1)

    # Initialize simulation parameters
    num_of_iter = int(simulation_time / time_step)
    time_array = np.arange(0, simulation_time, time_step)

    # Initialize arrays to store positions, velocities, and accelerations of all cars over time
    cars = np.zeros((num_of_iter, N, 3))
    cars_original = np.zeros((num_of_iter, N, 3))

    # Initial conditions for car 0 (leader)
    X = np.concatenate((del_xi, del_vi, del_ai), axis=1).transpose()
    X_current = X.copy()

    # attack_iter = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1500, 1501, 1502, 1503, 1504, 1505]

    # Kalman Filter Variables
    H_kalman = np.eye(3 * (N-1))
    Q = np.eye(3 * (N-1)) * 1e-5  # Process noise covariance
    R = np.eye(3 * (N-1)) * 1e-5  # Measurement noise covariance
    P = np.eye(3 * (N-1)) * 1e-5  # Initial state covariance

    noise_deviation = 0.5

    car_0_position = xi[0]
    car_0_velocity = vi[0]
    car_0_acceleration = ai[0]

    
    A_matrix, B_matrix = get_matrices(N, K, B, H, I)

    

    # Simulation loop
    for iter in range(num_of_iter):
        U = get_U(N, K, I)

        X_cap = np.matmul(A_matrix, X_current) + np.matmul(B_matrix, U)
  
        # if iter in attack_iter:
        #     noise = np.random.normal(0, 200, X_cap.shape)
        # else:
        #     noise = np.random.normal(0, noise_deviation, X_cap.shape)


        # Kalman filter measurement: here we assume the measurement Z to be the current state
        Z = X_cap.copy() 
        Z = Z + 10

        # Apply Kalman filter
        X_new, P = kalman_filter(X_current, P, A_matrix, B_matrix, U, H_kalman, Q, R, Z)


        def compare_with_threshold(matrix1, matrix2, threshold):
            # Compute the absolute difference between the matrices
            difference = np.abs(matrix1 - matrix2)
            
            # Check if the difference exceeds the threshold
            result = np.any(difference > threshold)
            return result

        if compare_with_threshold(Z, X_new, 50):
            print(f"Attack detected at iteration {iter}.")

        
        # Update X_current with Kalman filter output
        X_cap = X_new.copy()
        X_current = X_current + time_step * X_cap

        X_noisy = X_current + time_step * Z

        plot_data = X_current.copy().reshape(-1, N-1)
        plot_noisy = X_noisy.copy().reshape(-1, N-1)

        car_0_position += car_0_velocity * time_step + 0.5 * car_0_acceleration * time_step**2
        car_0_velocity += car_0_acceleration * time_step


        # Update the positions, velocities, and accelerations in the storage array
        for car in range(N):
            if car == 0:
                cars[iter][car][0] = car_0_position
                cars[iter][car][1] = car_0_velocity
                cars[iter][car][2] = car_0_acceleration
            else:
                cars[iter][car][0] = car_0_position + plot_data[0][N-car-1]  # Position
                cars[iter][car][1] = car_0_velocity + plot_data[1][N-car-1]  # Velocity
                cars[iter][car][2] = car_0_acceleration + plot_data[2][N-car-1]  # Acceleration

        # Update the positions, velocities, and accelerations in the storage array
        for car in range(N):
            if car == 0:
                cars_original[iter][car][0] = car_0_position
                cars_original[iter][car][1] = car_0_velocity
                cars_original[iter][car][2] = car_0_acceleration
            else:
                cars_original[iter][car][0] = car_0_position + plot_noisy[0][N-car-1]  # Position
                cars_original[iter][car][1] = car_0_velocity + plot_noisy[1][N-car-1]  # Velocity
                cars_original[iter][car][2] = car_0_acceleration + plot_noisy[2][N-car-1]  # Acceleration

    # Plot for correct

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 0], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Position of Cars in Platoon over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_positions.png")

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 1], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity of Cars in Platoon over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_velocities.png")

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 2], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s2)")
    plt.title("Acceleration of Cars in Platoon over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_accelerations.png")


    # Plot for noisy

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars_original[:, car, 0], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Position of Cars in Platoon over Time (Noisy)")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_positions_noisy.png")

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars_original[:, car, 1], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity of Cars in Platoon over Time (Noisy)")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_velocities_noisy.png")

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars_original[:, car, 2], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s2)")
    plt.title("Acceleration of Cars in Platoon over Time (Noisy)")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_accelerations_noisy.png")

    return cars

# Example usage
N = 5  # Number of cars, you can increase this

# Initial conditions (generalize for N cars)
li = np.array([4, 4.4, 3.8, 5.2, 4.4, 3.8, 4])
xi = np.array([0, -8, -20, -40, -80, -100, -120])
vi = np.array([25, 27.8, 22.2, 19.4, 27.8, 22.2, 27.8])
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

simulate_platoon(N, I, li[:N], xi[:N], vi[:N], ai[:N], desired_distance[:N], C, K, B, H, tau)