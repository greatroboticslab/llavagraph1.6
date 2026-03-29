import numpy as np
import matplotlib.pyplot as plt
import re


def simulate_noise_stabilization(filename):
    # 1. LOAD DATA
    raw_d = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('D:'):
                match = re.search(r'D:(-?\d+)', line)
                if match: raw_d.append(int(match.group(1)))

    y = np.array(raw_d)
    fs = 1000.0
    dt = 1.0 / fs
    time = np.arange(len(y)) / fs

    # Target is the average (trying to stay perfectly still)
    target = np.mean(y)

    # 2. PID SIMULATION (Active Cancellation)
    actual_pos = np.zeros_like(time)
    actual_pos[0] = y[0]
    integral = 0
    kp, ki = 0.5, 40.0  # High Integral gain to kill low-freq drift

    for i in range(1, len(time)):
        # Measure error relative to the noise in your file
        error = target - (np.round(actual_pos[i - 1]) + (y[i] - target))
        integral += error * dt
        v = (kp * error) + (ki * integral)
        v = np.clip(v, -10, 10)

        # Physics update
        actual_pos[i] = actual_pos[i - 1] + (v / 0.02) * dt

    # 3. RESULTS & PLOT
    raw_std = np.std(y)
    new_std = np.std(actual_pos)

    plt.figure(figsize=(12, 5))
    plt.step(time[:1000], y[:1000], 'r', alpha=0.3, label=f'Raw Noise (StdDev: {raw_std:.3f})')
    plt.plot(time[:1000], actual_pos[:1000], 'b', label=f'PID Stabilized (StdDev: {new_std:.3f})')
    plt.axhline(target, color='k', linestyle='--', label='Goal')
    plt.title(f"Active Noise Stabilization Simulation: {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("D-counts")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()


# Run it
simulate_noise_stabilization('14_Run2_1.txt')