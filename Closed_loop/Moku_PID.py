# Connect Moku:Go and power on the Piezo Amplifier.
# Run the Python script and enter the Moku's IP address.
# Crucial: Ensure the piezo is at its "resting" position. Your process_raw.py used a relative baseline—in a closed loop, the PID will treat the current sensor value as the starting point.

# Baseline: Ensure the piezo is at zero before clicking Start.
# Units: The script converts your target nm into the fractional D units the Moku expects.
# The "Square" Fix: If the motion still looks square, double your Ki value. The Integral term is specifically designed to eliminate steady-state error and smooth out digitized steps.

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import time
from moku.instruments import PIDController


class MokuPrecisionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Moku:Go Precision Piezo Control v2.0")
        self.root.geometry("700x650")

        # --- Variables ---
        self.wavelength = tk.DoubleVar(value=632.991372)
        self.moku_ip = tk.StringVar(value="192.168.###.###")
        self.kp, self.ki, self.kd = tk.DoubleVar(value=0.5), tk.DoubleVar(value=50.0), tk.DoubleVar(value=0.001)
        self.freq, self.amp_nm = tk.DoubleVar(value=1.0), tk.DoubleVar(value=400.0)
        self.wave_type = tk.StringVar(value="Sine")

        # Monitor Variables
        self.current_error = tk.StringVar(value="0.0 nm")
        self.is_running = False
        self.instrument = None

        self.setup_ui()

    def setup_ui(self):
        # Menus
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        pid_menu = tk.Menu(menubar, tearoff=0)
        pid_menu.add_command(label="Adjust Gains", command=self.open_pid_window)
        menubar.add_cascade(label="PID Tuning", menu=pid_menu)

        cal_menu = tk.Menu(menubar, tearoff=0)
        cal_menu.add_command(label="Wavelength Setup", command=self.open_cal_window)
        menubar.add_cascade(label="Calibration", menu=cal_menu)

        # Main Dashboard
        main = tk.Frame(self.root, padx=30, pady=20)
        main.pack(fill="both", expand=True)

        tk.Label(main, text="Target Amplitude (nm)").pack(anchor="w")
        tk.Scale(main, from_=0, to=1500, orient="horizontal", variable=self.amp_nm).pack(fill="x")

        tk.Label(main, text="Frequency (Hz)").pack(anchor="w", pady=(10, 0))
        tk.Scale(main, from_=0.1, to=5.0, resolution=0.1, orient="horizontal", variable=self.freq).pack(fill="x")

        # --- NEW: Data Monitor ---
        monitor_frame = tk.LabelFrame(main, text="Live Tracking Monitor", padx=10, pady=10)
        monitor_frame.pack(fill="x", pady=20)
        tk.Label(monitor_frame, text="Current Tracking Error:").grid(row=0, column=0)
        tk.Label(monitor_frame, textvariable=self.current_error, font=("Courier", 12, "bold"), fg="blue").grid(row=0,
                                                                                                               column=1,
                                                                                                               padx=10)

        self.status = tk.Label(main, text="Status: IDLE", font=("Arial", 12, "bold"))
        self.status.pack(pady=10)

        tk.Button(main, text="START CLOSED-LOOP", bg="#28a745", fg="white", font=("Arial", 12, "bold"), height=2,
                  command=self.start_system).pack(fill="x")
        tk.Button(main, text="STOP SYSTEM", bg="#dc3545", fg="white", command=self.stop_system).pack(fill="x", pady=5)

    def open_cal_window(self):
        win = tk.Toplevel(self.root)
        win.title("Calibration")
        tk.Label(win, text="Wavelength (nm):").pack(padx=20, pady=5)
        tk.Entry(win, textvariable=self.wavelength).pack(padx=20, pady=5)

    def open_pid_window(self):
        win = tk.Toplevel(self.root)
        win.title("PID Tuning")
        for l, v in [("Kp", self.kp), ("Ki", self.ki), ("Kd", self.kd)]:
            f = tk.Frame(win);
            f.pack(pady=5)
            tk.Label(f, text=l, width=5).pack(side="left")
            tk.Entry(f, textvariable=v).pack(side="right")
        tk.Button(win, text="Update Hardware", command=self.apply_hardware).pack(pady=10)

    def apply_hardware(self):
        if self.instrument:
            self.instrument.set_by_gain(channel=1, overall_gain=0, prop_gain=self.kp.get(), int_gain=self.ki.get(),
                                        diff_gain=self.kd.get())

    def start_system(self):
        try:
            self.instrument = PIDController(self.moku_ip.get(), force_connect=True)
            self.instrument.set_frontend(1, coupling='DC', impedance='1MOhm', range='10Vpp')
            self.instrument.set_output_limit(channel=1, lower=0, upper=5)
            self.apply_hardware()
            self.instrument.enable_control(channel=1, enable=True)
            self.is_running = True
            self.status.config(text="Status: RUNNING", fg="green")
            threading.Thread(target=self.waveform_engine, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def stop_system(self):
        self.is_running = False
        if self.instrument: self.instrument.set_input_offset(channel=1, offset=0)
        self.status.config(text="Status: STOPPED", fg="red")

    def waveform_engine(self):
        start_t = time.time()
        while self.is_running:
            elapsed = time.time() - start_t
            nm_per_step = self.wavelength.get() / 8.0

            # 1. Calculate Target
            desired_nm = (self.amp_nm.get() / 2) * np.sin(2 * np.pi * self.freq.get() * elapsed) + (
                        self.amp_nm.get() / 2)
            target_d = desired_nm / nm_per_step
            self.instrument.set_input_offset(channel=1, offset=target_d)

            # 2. Update Monitor (Read actual position from Moku)
            try:
                # We probe the input to see the actual sensor value
                data = self.instrument.get_control_loop_data()
                actual_d = data['input']
                error = (target_d - actual_d) * nm_per_step
                self.current_error.set(f"{error:.2f} nm")
            except:
                pass

            time.sleep(0.01)


if __name__ == "__main__":
    root = tk.Tk()
    app = MokuPrecisionSystem(root)
    root.mainloop()