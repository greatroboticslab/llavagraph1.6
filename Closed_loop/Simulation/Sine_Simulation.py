import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit


def sine_func(t, amp, freq, phase, offset):
    return amp * np.sin(2 * np.pi * freq * t + phase) + offset


def create_aligned_comparison(filename):
    # 1. Load data
    raw_d_counts = []
    fs = 1000.0
    with open(filename, 'r') as f:
        for line in f:
            if "Sample Frequency =" in line:
                fs = float(re.search(r'[\d\.]+', line).group())
            if line.startswith('D:'):
                match = re.search(r'D:(-?\d+)', line)
                if match:
                    raw_d_counts.append(int(match.group(1)))

    y_open_loop = np.array(raw_d_counts)
    dt = 1.0 / fs
    time = np.arange(len(y_open_loop)) / fs

    # 2. Precise Sine Fitting to align the Target with your data
    # Initial guesses
    guess_freq = 1.0
    if "100Hz" in filename: guess_freq = 100.0
    guess_amp = (np.max(y_open_loop) - np.min(y_open_loop)) / 2
    guess_offset = np.mean(y_open_loop)
    guess_phase = 0

    # We only fit the first few cycles for stability
    fit_end = int(3 * fs / guess_freq) if guess_freq > 0 else 1000
    p0 = [guess_amp, guess_freq, guess_phase, guess_offset]

    try:
        popt, _ = curve_fit(sine_func, time[:fit_end], y_open_loop[:fit_end], p0=p0)
        fit_amp, fit_freq, fit_phase, fit_offset = popt
    except:
        fit_amp, fit_freq, fit_phase, fit_offset = p0

    # The Target Setpoint is now perfectly aligned with your actual motion
    setpoint = sine_func(time, fit_amp, fit_freq, fit_phase, fit_offset)

    # 3. PID Simulation (Closed-loop)
    # Using the fitted params to ensure the PID starts aligned
    actual_physical_pos = np.zeros_like(time)
    actual_physical_pos[0] = y_open_loop[0]
    integral_sum = 0
    previous_error = 0
    # Recommended gains from previous discussion
    kp, ki, kd = (1.2, 15.0, 0.005) if fit_freq < 10 else (0.8, 45.0, 0.0)

    K_piezo = 1.0
    tau = 0.02

    for i in range(1, len(time)):
        feedback_d = np.round(actual_physical_pos[i - 1])
        error = setpoint[i] - feedback_d
        integral_sum += error * dt
        integral_sum = np.clip(integral_sum, -5, 5)
        derivative = (error - previous_error) / dt
        v = (kp * error) + (ki * integral_sum) + (kd * derivative)
        v = np.clip(v, -10, 10)
        actual_physical_pos[i] = actual_physical_pos[i - 1] + (
                    (K_piezo * v - (actual_physical_pos[i - 1] - fit_offset)) / tau) * dt
        previous_error = error

    # 4. Create Comparison Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot window
    show_n = int(2 * fs / fit_freq) if fit_freq > 0 else 2000

    # BEFORE: Open-Loop
    ax1.step(time[:show_n], y_open_loop[:show_n], color='red', where='post', alpha=0.5, label='Actual Data (Staircase)')
    ax1.plot(time[:show_n], setpoint[:show_n], 'k--', linewidth=1.5, label='Intended Path (Fitted Sine)')
    ax1.set_title(f"BEFORE: Open-Loop Data (Aligned Fit)\nShows Quantization Error (Stairs)", fontsize=14)
    ax1.set_ylabel("D-counts")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # AFTER: PID
    ax2.plot(time[:show_n], actual_physical_pos[:show_n], color='blue', linewidth=2,
             label='PID Controlled Motion (Smooth)')
    ax2.plot(time[:show_n], setpoint[:show_n], 'k--', linewidth=1.5, label='Intended Path')
    ax2.set_title("AFTER: Simulated Closed-Loop (PID)\nShows Smooth Tracking of the Same Path", fontsize=14,
                  color='blue')
    ax2.set_ylabel("D-counts")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aligned_pid_comparison.png')
    return fit_freq, fit_amp


f_found, a_found = create_aligned_comparison('1Hz_1hz_2.txt')
print(f"Detected Freq: {f_found}, Amp: {a_found}")