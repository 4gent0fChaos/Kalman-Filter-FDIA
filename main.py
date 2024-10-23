import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import matplotlib
matplotlib.use('TkAgg')
from utils import *


# Generalized function to handle N cars
def simulate_platoon(N, I, li, xi, vi, ai, desired_distance, C, K, B, H, tau, simulation_time=10, time_step=0.005):


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


    # Kalman Filter Variables
    H_kalman = np.eye(3 * (N-1))
    Q = np.eye(3 * (N-1)) * 0.5  # Process noise covariance
    R = np.eye(3 * (N-1)) * 1e-5  # Measurement noise covariance
    P = np.eye(3 * (N-1)) * 0.5  # Initial state covariance

    SNR = 50

    car_0_position = xi[0]
    car_0_velocity = vi[0]
    car_0_acceleration = ai[0]

    
    A_matrix, B_matrix = get_matrices(N, K, B, H, I, C, tau)

    attack_val = 0

    

    # Simulation loop
    for iter in range(num_of_iter):
        current_time = iter * time_step
        U = get_U(N, K, I, tau, li, desired_distance) 

        Z = np.matmul(A_matrix, X_current) + np.matmul(B_matrix, U)

        Z_noisy = add_noise_based_on_snr(Z, SNR) 
        X_noisy = add_noise_based_on_snr(X_current, SNR)         

        plot_noisy = X_noisy.copy().reshape(-1, N-1)

        
        # Apply Kalman filter
        X_new, P = kalman_filter(X_noisy, P, A_matrix, B_matrix, U, H_kalman, Q, R, Z_noisy)
        
        # Update X_current with Kalman filter output
        X_current = X_current + time_step * X_new

        plot_data = X_current.copy().reshape(-1, N-1)

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
    plt.title(f"Position of Cars in Platoon over Time ({SNR}db)")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_positions.png")

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 1], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"Velocity of Cars in Platoon over Time ({SNR}db)")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_velocities.png")

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 2], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s2)")
    plt.title(f"Acceleration of Cars in Platoon over Time ({SNR}db)")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_accelerations.png")


    # Plot for noisy

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars_original[:, car, 0], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(f"Position of Cars in Platoon over Time (Noisy) ({SNR}db)")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_positions_noisy.png")

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars_original[:, car, 1], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"Velocity of Cars in Platoon over Time (Noisy) ({SNR}db)")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_velocities_noisy.png")

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars_original[:, car, 2], label=f"Car {car} (Follower)", linestyle='-')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s2)")
    plt.title(f"Acceleration of Cars in Platoon over Time (Noisy) ({SNR}db)")
    plt.legend()
    plt.grid(True)
    plt.savefig("platoon_accelerations_noisy.png")

    return cars

# Example usage
N = 7  # Number of cars, you can increase this

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

simulate_platoon(N, I[:N], li[:N], xi[:N], vi[:N], ai[:N], desired_distance[:N], C[:N], K, B, H, tau)