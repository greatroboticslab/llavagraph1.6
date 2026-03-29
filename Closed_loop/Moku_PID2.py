# Moku IP : 192.168.73.1

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import time
import os

# Try to import moku; if it fails, the script will warn you
try:
    from moku.instruments import PIDController
except ImportError:
    print("Error: moku library not found. Please run 'pip install moku'")


class UltraPiezoController:
    def __init__(self, root):
        self.root = root
        self.root.title("Moku:Go Precision Dashboard")
        self.root.geometry("600x700")

        # --- CONFIG FILE FOR CONVENIENCE ---
        self.config_file = "moku_config.txt"
        saved_ip = self.load_ip()

        # --- Variables ---
        self.moku_ip = tk.StringVar(value=saved_ip)
        self.wavelength = tk.DoubleVar(value=632.991372)
        self.kp = tk.DoubleVar(value=1.0)
        self.ki = tk.DoubleVar(value=150.0)
        self.kd = tk.DoubleVar(value=0.005)

        self.freq = tk.DoubleVar(value=1.0)
        self.amp_nm = tk.DoubleVar(value=500.0)
        self.wave_type = tk.StringVar(value="Sine")

        self.is_running = False
        self.instrument = None
        self.error_val = tk.StringVar(value="0.00 nm")

        self.setup_ui()

    def load_ip(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                return f.read().strip()
        return "192.168.###.###"

    def save_ip(self, ip):
        with open(self.config_file, "w") as f:
            f.write(ip)

    def setup_ui(self):
        # --- 1. IP CONNECTION PANEL (TOP & CLEAR) ---
        conn_frame = tk.LabelFrame(self.root, text=" STEP 1: Connect to Moku:Go ", padx=10, pady=10, fg="blue")
        conn_frame.pack(fill="x", padx=15, pady=10)

        tk.Label(conn_frame, text="Enter IP Address:").pack(side="left", padx=5)
        self.ip_entry = tk.Entry(conn_frame, textvariable=self.moku_ip, width=25, font=("Arial", 12))
        self.ip_entry.pack(side="left", padx=5)

        self.btn_connect = tk.Button(conn_frame, text="CONNECT", bg="green", fg="white",
                                     font=("Arial", 10, "bold"), command=self.connect_hardware)
        self.btn_connect.pack(side="left", padx=10)

        # --- 2. WAVEFORM SETTINGS ---
        wave_frame = tk.LabelFrame(self.root, text=" STEP 2: Waveform Settings ", padx=10, pady=10)
        wave_frame.pack(fill="x", padx=15, pady=5)

        tk.Label(wave_frame, text="Target Amplitude (nm):").pack(anchor="w")
        tk.Scale(wave_frame, from_=0, to=2000, orient="horizontal", variable=self.amp_nm).pack(fill="x")

        tk.Label(wave_frame, text="Frequency (Hz):").pack(anchor="w", pady=(10, 0))
        tk.Scale(wave_frame, from_=0.1, to=10, resolution=0.1, orient="horizontal", variable=self.freq).pack(fill="x")

        # --- 3. PID TUNING (To fix the Square look) ---
        pid_frame = tk.LabelFrame(self.root, text=" STEP 3: Tuning (Fix Square Look) ", padx=10, pady=10)
        pid_frame.pack(fill="x", padx=15, pady=5)

        for i, (txt, var) in enumerate([("Kp", self.kp), ("Ki", self.ki), ("Kd", self.kd)]):
            tk.Label(pid_frame, text=txt).grid(row=0, column=i * 2, padx=5)
            tk.Entry(pid_frame, textvariable=var, width=8).grid(row=0, column=i * 2 + 1, padx=5)

        tk.Button(pid_frame, text="APPLY GAINS LIVE", command=self.sync_gains).grid(row=1, column=0, columnspan=6,
                                                                                    pady=10)

        # --- 4. DATA MONITOR ---
        mon_frame = tk.LabelFrame(self.root, text=" Tracking Performance ", padx=10, pady=10)
        mon_frame.pack(fill="x", padx=15, pady=5)
        tk.Label(mon_frame, text="Current Error:").pack(side="left")
        tk.Label(mon_frame, textvariable=self.error_val, font=("Arial", 16, "bold"), fg="red").pack(side="left",
                                                                                                    padx=20)

        # --- 5. CONTROL BUTTONS ---
        self.btn_run = tk.Button(self.root, text="START CLOSED-LOOP", bg="#28a745", fg="white",
                                 font=("Arial", 14, "bold"), height=2, state="disabled", command=self.start_loop)
        self.btn_run.pack(fill="x", padx=15, pady=10)

        tk.Button(self.root, text="EMERGENCY STOP", bg="red", fg="white", command=self.stop_loop).pack(fill="x",
                                                                                                       padx=15)

    # --- LOGIC ---
    def connect_hardware(self):
        ip = self.moku_ip.get()
        try:
            self.instrument = PIDController(ip, force_connect=True)
            self.save_ip(ip)  # Save for next time!
            self.instrument.set_frontend(1, coupling='DC', impedance='1MOhm', range='10Vpp')
            self.instrument.set_output_limit(channel=1, lower=0, upper=5)
            self.btn_run.config(state="normal")
            self.btn_connect.config(text="CONNECTED!", bg="gray")
            messagebox.showinfo("Success", f"Connected to Moku at {ip}")
        except Exception as e:
            messagebox.showerror("Connection Error", f"Check IP Address and Wi-Fi/USB:\n{e}")

    def sync_gains(self):
        if self.instrument:
            self.instrument.set_by_gain(channel=1, overall_gain=0,
                                        prop_gain=self.kp.get(),
                                        int_gain=self.ki.get(),
                                        diff_gain=self.kd.get())

    def start_loop(self):
        self.is_running = True
        self.sync_gains()
        self.instrument.enable_control(channel=1, enable=True)
        threading.Thread(target=self.engine, daemon=True).start()

    def stop_loop(self):
        self.is_running = False
        if self.instrument:
            self.instrument.set_input_offset(channel=1, offset=0)
        messagebox.showwarning("System Stopped", "Closed-loop control deactivated.")

    def engine(self):
        start_t = time.time()
        nm_per_d = self.wavelength.get() / 8.0

        while self.is_running:
            t = time.time() - start_t

            # Create Target
            target_nm = (self.amp_nm.get() / 2) * np.sin(2 * np.pi * self.freq.get() * t) + (self.amp_nm.get() / 2)
            target_d = target_nm / nm_per_d

            # Push Target to Moku
            self.instrument.set_input_offset(channel=1, offset=target_d)

            # Monitor Error every 100ms
            if int(t * 10) % 2 == 0:
                try:
                    actual_d = self.instrument.get_control_loop_data()['input']
                    err = (target_d - actual_d) * nm_per_d
                    self.error_val.set(f"{err:.2f} nm")
                except:
                    pass

            time.sleep(0.01)


if __name__ == "__main__":
    root = tk.Tk()
    app = UltraPiezoController(root)
    root.mainloop()