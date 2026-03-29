import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit
from scipy.signal import square

# Square wave model to align with data
def square_model(t, amp, freq, phase, offset):
    return amp * square(2 * np.pi * freq * t + phase) + offset

def run_square_pid(filename):
    # 1. LOAD DATA
    raw_d = []
    fs = 1000.0
    with open(filename, 'r') as f:
        for line in f:
            if "Sample Frequency =" in line:
                fs = float(re.search(r'[\d\.]+', line).group())
            if line.startswith('D:'):
                match = re.search(r'D:(-?\d+)', line)
                if match: raw_d.append(int(match.group(1)))

    y = np.array(raw_d)
    time = np.arange(len(y)) / fs
    dt = 1.0 / fs

    # 2. FIT SQUARE WAVE (Align phase and height)
    # Guess parameters for solver
    guess_amp = (np.max(y) - np.min(y)) / 2
    guess_freq = 1.0 # Default
    if "100Hz" in filename: guess_freq = 100.0
    elif "300Hz" in filename: guess_freq = 300.0

    p0 = [guess_amp, guess_freq, 0, np.mean(y)]
    popt, _ = curve_fit(square_model, time[:2000], y[:2000], p0=p0)
    fit_amp, fit_freq, fit_phase, fit_offset = popt

    # 3. PID SIMULATION
    setpoint = square_model(time, fit_amp, fit_freq, fit_phase, fit_offset)
    actual_pos = np.zeros_like(time)
    actual_pos[0] = y[0]
    integral, prev_err = 0, 0
    kp, ki = 0.6, 25.0 # PID Gains for square tracking

    for i in range(1, len(time)):
        err = setpoint[i] - np.round(actual_pos[i-1])
        integral += err * dt
        v = (kp * err) + (ki * integral)
        v = np.clip(v, -10, 10)
        # Fast piezo model (tau=0.01)
        actual_pos[i] = actual_pos[i-1] + ((1.0*v - (actual_pos[i-1]-fit_offset))/0.01)*dt

    # 4. PLOT
    plt.figure(figsize=(12, 6))
    window = int(4 * fs / fit_freq) if fit_freq > 0 else 1000
    plt.step(time[:window], y[:window], 'r', alpha=0.3, label='BEFORE (Raw Stairs)')
    plt.plot(time[:window], setpoint[:window], 'k--', label='ALIGNED TARGET')
    plt.plot(time[:window], actual_pos[:window], 'b', linewidth=2, label='AFTER (PID Smooth)')
    plt.title(f"Square Wave PID Tracking: {filename}")
    plt.legend()
    plt.show()

# Run for one of your files
run_square_pid('300Hz_300Hz_9.txt')