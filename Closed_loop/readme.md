# Piezo Actuator Closed-Loop Control

This repository contains Python-based simulations designed to demonstrate the transition of a Piezo Actuator system from Open-Loop (raw, quantized motion) to Closed-Loop (PID) control.

The project uses real-world data from a Moku:Go/Moku:Pro (D-counts) to model how a Proportional-Integral-Derivative (PID) controller can eliminate quantization "stairs" and provide smooth, nanometer-scale precision.

## Project Overview

Piezo actuators often suffer from quantization error at small scales, where the sensor jumps between discrete digital values (D-counts). This project provides a pipeline to:
- Analyze raw sensor data from .txt or .csv files.
- Align ideal mathematical targets (Phase-Matching) to existing data.
- Simulate a PID controller's response to smooth out motion and reduce variance.

## Idea

The fundamental principle of this system is that we do not simply "send a wave" to the piezo. Instead, we constantly calculate how far the piezo is from where it *should* be and generate a corrective voltage.

### 1. The Measurement (Feedback)

First, you get the Actual Motion from the software (the Moku sensor data). In our case, these are the D-counts.
- Actual Motion: The current position of the piezo as reported by the sensor.
- Desired Motion: The "Setpoint" or input (the perfect Sine, Square, or Static line).


### 2. The Error Calculation ($e$)

The "Error" is simply the difference between what you want and what you have at any specific millisecond:
$$e(t) = \text{Setpoint}(t) - \text{Actual}(t)$$

- If $e = 0$, the piezo is perfectly on target.
- If $e$ is positive, the piezo is too low.
- If $e$ is negative, the piezo is too high.

### 3. Multiplying by PID Gains
We multiply this error by three specific gains ($K_p$, $K_i$, $K_d$) to calculate the necessary correction. The sum of these three components determines the **Control Output Voltage ($V_{out}$)**:

$$V_{out}(t) = K_p e(t) + K_i \int_{0}^{t} e(\tau) d\tau + K_d \frac{de(t)}{dt}$$

* **Proportional ($K_p$):** Provides an immediate push based on the current error.
* **Integral ($K_i$):** Adds up past errors to "fill in" the quantization stairs and remove steady-state drift.
* **Derivative ($K_d$):** Predicts future error to dampen the motion and prevent overshoot.

### 4. The Output: Voltage ($V_{out}$)

The final result of the formula is not a position; it is a Voltage.

$$V_{out} = (P \text{ result}) + (I \text{ result}) + (D \text{ result})$$

This voltage is sent to the Moku Waveform Generator, which drives the Piezo Actuator. If the error was positive, the PID increases the voltage to push the piezo up.

## Pipeline

**The Software Implementation (Python)**
In the code I gave you, the "Loop" happens inside the for line:

```
for i in range(1, len(time)):
    # 1. Compare Desired vs Actual
    error = setpoint[i] - feedback_from_piezo
    
    # 2. Accumulate Integral (The "I")
    integral += error * dt
    
    # 3. Calculate Voltage Output
    voltage_out = (kp * error) + (ki * integral) + (kd * (error - prev_err)/dt)
    
    # 4. Update the "Physical" Piezo position based on that Voltage
    feedback_from_piezo = simulate_piezo_physics(voltage_out)
```


**The Hardware Implementation (Moku:Go)**

To do this in real life using the Moku:
- Input 1: Connect your Piezo sensor (D-counts/Voltage) to Moku Input 1. Raw sensor data.
- Control Matrix: Moku compares Input 1 to Setpoint.
- AUTOMATIC Gain Entry: Python API calculates and writes $K$ values to the PID engine.
- Output 1: Corrected voltage to the Piezo Amplifier.
- The Result: The Moku will perform the calculation (Error $\times$ Gains) thousands of times per second, automatically adjusting the voltage to keep the error at zero.

```
from moku.instruments import PIDController

# 1. Connect to your Moku:Go via IP
i = PIDController('192.168.1.15', force_connect=True)

# 2. Calculate gains based on your simulation results
# (Example: using the 120Hz sine wave results)
calculated_kp = 1.2
calculated_ki = 15.0

# 3. AUTOMATIC GAIN ENTRY: Push to hardware
i.set_control_loop_parameters(channel=1, prop_gain=calculated_kp, int_gain=calculated_ki)

print(f"Hardware Updated: Kp={calculated_kp}, Ki={calculated_ki}")
```

## Waveform (Sine, Square, Noise)

1. Sine Wave (Continuous Smoothing)
- Problem: The raw piezo motion appears as a "staircase" sine wave due to low-resolution feedback.
- PID Solution: The controller treats the staircase as a disturbance and forces the actuator to follow the "average" smooth curve.
- Result: A continuous, high-fidelity sine path.
<img width="1200" height="1000" alt="sineCode_Generated_Image copy" src="https://github.com/user-attachments/assets/58b0f08b-8ec6-42df-8b0a-1cfb242def23" />

2. Square Wave (Fast Step Response)
- Problem: Abrupt jumps cause the sensor to flicker between two digital bits (quantization noise) at the plateaus.
- PID Solution: Optimized $K_p$ and $K_d$ values allow for a fast rise time while the $K_i$ term holds the plateau at a sub-bit level.
- Result: Sharp transitions with stabilized, non-flickering plateaus.
<img width="1200" height="1000" alt="squareCode_Generated_Image" src="https://github.com/user-attachments/assets/3c85df04-a395-493e-a2d0-9836726ba7e0" />


3. Random Noise (Active Stabilization)
- Problem: The actuator "twitches" due to environmental noise or sensor jitter.
- PID Solution: The system is set to a fixed Setpoint (the mean). The PID acts as an active damper, applying counter-voltages to "fight" the noise.
- Result: Significant reduction in Standard Deviation ($\sigma$); the "spread" of the noise is typically reduced by 50–70%.
<img width="1200" height="1000" alt="randomCode_Generated_Image copy" src="https://github.com/user-attachments/assets/744e8890-bf18-4242-a7cb-76ff7f1f208b" />


### Technical Implementation
The simulation is built using the following stack:
- NumPy: High-speed array math for the control loop.
- SciPy: Curve fitting for phase-matching and FFT for frequency detection.
- Matplotlib: Multi-pane visualization of the control error.

### Suggested PID Controller Settings(based on simulation)

| Waveform | Frequency Range | Suggested $K_p$ | Suggested $K_i$ | Suggested $K_d$ |
| :--- | :--- | :--- | :--- | :--- |
| **Sine** | 1Hz - 100Hz | 1.2 | 15.0 | 0.005 |
| **Square** | 100Hz - 400Hz | 0.6 | 30.0 | 0.002 |
| **Noise** | DC / Static | 0.5 | 45.0 | 0.010 |

### Visualization
When running the scripts in `Simulation`, the output displays two panels:
- Top Panel (Before): The raw, jagged red line showing the quantization "stairs."
- Bottom Panel (After): The smooth blue line showing the PID-controlled physical motion tracking the ideal target.





