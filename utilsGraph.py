import matplotlib.pyplot as plt


def make_true_est_graph(trueF, estF, time_array):
    plt.figure(figsize=(14, 6))

    # First subplot for the True F
    plt.subplot(1, 2, 1)  # (rows, columns, index)
    plt.plot(time_array, trueF, label="True F", linewidth=2, linestyle="--", color="blue")
    plt.title("True F", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Unknown Input (F)", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)

    # Second subplot for the Estimated F
    plt.subplot(1, 2, 2)
    plt.plot(time_array, estF, label="Estimated F", linewidth=2, linestyle="-", color="red")
    plt.title("Estimated F", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Unknown Input (F)", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the figure
    plt.savefig("fig/TE_Difference.png")



def make_graph(N, time_array, cars, SNR):
    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 0], label=f"Car {car} (Follower)", linestyle='-')

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(f"Position of Cars in Platoon over Time (After filters) ({SNR}db)")

    plt.legend()
    plt.grid(True)
    plt.savefig("fig/platoon_positions.png")


    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 1], label=f"Car {car} (Follower)", linestyle='-')

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"Velocity of Cars in Platoon over Time (After filters) ({SNR}db)")

    plt.legend()
    plt.grid(True)
    plt.savefig("fig/platoon_velocities.png")
        

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 2], label=f"Car {car} (Follower)", linestyle='-')

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s2)")
    plt.title(f"Acceleration of Cars in Platoon over Time (After filters) ({SNR}db)")

    plt.legend()
    plt.grid(True)
    plt.savefig("fig/platoon_accelerations.png")


def make_graph_filterless(N, time_array, cars, SNR):
    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 0], label=f"Car {car} (Follower)", linestyle='-')

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(f"Position of Cars in Platoon over Time (Without filters) ({SNR}db)")

    plt.legend()
    plt.grid(True)
    plt.savefig("fig/platoon_positions_filterless.png")


    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 1], label=f"Car {car} (Follower)", linestyle='-')

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"Velocity of Cars in Platoon over Time (Without filters) ({SNR}db)")

    plt.legend()
    plt.grid(True)
    plt.savefig("fig/platoon_velocities_filterless.png")
        

    plt.figure(figsize=(10, 6))
    for car in range(N):
        plt.plot(time_array, cars[:, car, 2], label=f"Car {car} (Follower)", linestyle='-')

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s2)")
    plt.title(f"Acceleration of Cars in Platoon over Time (Without filters) ({SNR}db)")

    plt.legend()
    plt.grid(True)
    plt.savefig("fig/platoon_accelerations_filterless.png")


def make_graph_res(res, time_array):
    plt.figure(figsize=(14, 6))

    plt.plot(time_array, res, label="Residual", linewidth=2, linestyle="-", color="blue")
    plt.title("Residual vs Time", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Residual", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig("fig/res.png")