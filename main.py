import numpy as np
import matplotlib.pyplot as plt

# Kalman filter function
def kalman_filter(X, P, A, B, U, H, Q, R, Z):
    # Prediction step
    X_pred = np.dot(A, X) + np.dot(B, U)
    P_pred = np.dot(A, np.dot(P, A.T)) + Q
    
    # Update step
    S = np.dot(H, (np.dot(P_pred, H.T)+R))
    K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))

    X_new = X_pred + np.dot(K, (Z - np.dot(H, X_pred)))
    P_new = np.dot((np.eye(len(K)) - np.dot(K, H)), P_pred)

    return X_new, P_new

def get_matrices(n):
    Z_n_n = np.zeros((n, n))
    I_n_n = np.eye(n)

    Xi = Z_n_n.copy()
    Omega = Z_n_n.copy()
    Lambda = Z_n_n.copy()
    for i in range(n):
        for j in range(n):
            if (i == j):
                Xi[i][i] = -1 * C[n-1-i] * K[n-1-i] / tau[n-1-i]
                Omega[i][i] = -1 * C[n-1-i] * B[n-1-i] / tau[n-1-i]
                Lambda[i][i] = -1 * (1 + C[n-1-i] * H[n-1-i]) / tau[n-1-i]
            else:
                Xi[i][j] = K[n-1-i] / tau[n-1-i] * get_S(i, j)
                Omega[i][j] = B[n-1-i] / tau[n-1-i] * get_S(i, j)
                Lambda[i][j] = H[n-1-i] / tau[n-1-i] * get_S(i, j)
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
    return (A_matrix, B_matrix)

def get_U(n):
    U = np.zeros((n, 1))  
    for k in range(n):
        sum_distances = 0
        for j in I[n-k]:  
            sum_distances += get_desired_distance(n-k, j)
        
        U[k] = K[k] / tau[k] * sum_distances  
    return U

def get_S(i, j):
    if i in I[j]:
        return 1
    return 0

def sgn(i, j):
    if i > j:
        return 1
    return -1

def get_desired_distance(i, j):
    d = 0
    for k in range(min([i, j]), max([i, j])):
        d += li[k] + desired_distance[k]
    return -1 * sgn(i, j) * d

# PLATOON OF 4 CARS(0, 1, 2, 3; 0 LEADER)
I = [[], [0], [0, 1], [0, 2]]

li = np.array([4, 4.4, 3.8, 5.2])
xi = np.array([0, -8, -20, -40])
vi = np.array([25, 27.8, 22.2, 19.4])
ai = np.array([0, 2, 3, 2])
desired_distance = np.array([3, 4, 4])

# Currently I'm calculating delta manually for testing
del_xi = np.array([[-40, -20, -3]])
del_vi = np.array([[-5.6, -2.8, 2.8]])
del_ai = np.array([[2, 3, 2]])

# Simulation parameters
simulation_time = 10
time_step = 0.005
num_of_iter = int(simulation_time/time_step)

# Initialize arrays to store positions of all cars over time
time_array = np.arange(0, simulation_time, time_step)
car_0 = np.zeros((num_of_iter, 3))  # Car 0 (leader)
car_1 = np.zeros((num_of_iter, 3))  # Car 1 (follower)
car_2 = np.zeros((num_of_iter, 3))  # Car 2 (follower)
car_3 = np.zeros((num_of_iter, 3))  # Car 3 (follower)

# Initial conditions for car 0 (leader)
car_0_position = xi[0]
car_0_velocity = vi[0]
car_0_acceleration = ai[0]

X = np.concatenate((del_xi, del_vi, del_ai), axis=1).transpose()
X_current = X.copy()

K = [3, 3, 3]
B = [5, 5, 5]
H = [1, 1, 1]
tau = [0.5, 0.5, 0.5]

n = 3
C = [len(I[1]), len(I[2]), len(I[3])]

A_matrix, B_matrix = get_matrices(n)

# Kalman Filter Variables
H_kalman = np.eye(3 * n)

Q = np.eye(3 * n) * 1e-5  # Process noise covariance
R = np.eye(3 * n) * 1e-5  # Measurement noise covariance

P = np.eye(3 * n) * 1e-5

# Updation loop
for iter in range(num_of_iter):
    # Update car 0 position based on its velocity and acceleration
    car_0_position += car_0_velocity * time_step + 0.5 * car_0_acceleration * time_step**2
    car_0_velocity += car_0_acceleration * time_step

    U = get_U(n)

    X_cap = np.matmul(A_matrix, X_current) + np.matmul(B_matrix, U)     # Currently using this as measurements

    # Kalman filter measurement: here we assume the measurement Z to be the current state
    Z = X_cap.copy()

    # Apply Kalman filter
    X_new, P = kalman_filter(X_current, P, A_matrix, B_matrix, U, H_kalman, Q, R, Z)
    
    # Update X_current with Kalman filter output
    X_cap = X_new.copy()

    X_current = X_current + time_step * X_cap

    # Putting values in array for graph
    car_0[iter][0] = car_0_position
    car_0[iter][1] = car_0_velocity
    car_0[iter][2] = car_0_acceleration
    car_1[iter][0] = car_0_position + X_current[2, 0]
    car_1[iter][1] = car_0_velocity + X_current[5, 0]
    car_1[iter][2] = X_current[8, 0]
    car_2[iter][0] = car_0_position + X_current[1, 0]   
    car_2[iter][1] = car_0_velocity + X_current[4, 0]   
    car_2[iter][2] = X_current[7, 0]
    car_3[iter][0] = car_0_position + X_current[0, 0]
    car_3[iter][1] = car_0_velocity + X_current[3, 0]
    car_3[iter][2] = X_current[6, 0]

    # print(f"Time {iter * time_step:.1f}s: X = \n{X_current}")
    # print(f"Time {iter * time_step:.1f}s: X = \n{X_new}")

    



plt.figure(figsize=(10, 6))
plt.plot(time_array, car_0.transpose()[0], label="Car 0 (Leader)", linestyle='-')
plt.plot(time_array, car_1.transpose()[0], label="Car 1 (Follower)", linestyle='-')
plt.plot(time_array, car_2.transpose()[0], label="Car 2 (Follower)", linestyle='-')
plt.plot(time_array, car_3.transpose()[0], label="Car 3 (Follower)", linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Position of Cars in Platoon over Time")
plt.legend()
plt.grid(True)


plt.savefig("platoon_positions.png")

plt.figure(figsize=(10, 6))
plt.plot(time_array, car_0.transpose()[1], label="Car 0 (Leader)", linestyle='-')
plt.plot(time_array, car_1.transpose()[1], label="Car 1 (Follower)", linestyle='-')
plt.plot(time_array, car_2.transpose()[1], label="Car 2 (Follower)", linestyle='-')
plt.plot(time_array, car_3.transpose()[1], label="Car 3 (Follower)", linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Valocity of Cars in Platoon over Time")
plt.legend()
plt.grid(True)

plt.savefig("platoon_velocity.png")

plt.figure(figsize=(10, 6))
plt.plot(time_array, car_0.transpose()[2], label="Car 0 (Leader)", linestyle='-')
plt.plot(time_array, car_1.transpose()[2], label="Car 1 (Follower)", linestyle='-')
plt.plot(time_array, car_2.transpose()[2], label="Car 2 (Follower)", linestyle='-')
plt.plot(time_array, car_3.transpose()[2], label="Car 3 (Follower)", linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s2)")
plt.title("Acceleration of Cars in Platoon over Time")
plt.legend()
plt.grid(True)

plt.savefig("platoon_acceleration.png")

