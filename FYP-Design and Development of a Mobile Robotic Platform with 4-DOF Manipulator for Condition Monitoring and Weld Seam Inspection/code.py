import tkinter as tk
from tkinter import ttk, messagebox
import math
import threading
import time
import serial
import cv2
from PIL import Image, ImageTk
import bcrypt
import numpy as np
from ultralytics import YOLO
import torch

# --- New Imports for Dynamic Path Resolution and Video Recording ---
import os
import sys
import datetime

# Define CAMERA_SOURCE. Use 0 for default webcam, or a different index if multiple cameras are present.
# For a video file, replace 0 with the path to the video file (e.g., "path/to/your/video.mp4").
CAMERA_SOURCE = 1


class CBMHMI:
    def __init__(self, master):
        """
        Initializes the CBM_HMI application.
        Sets up the main window, styling, and initializes variables for
        robot arm control, sensor data, serial communication, camera,
        YOLOv8 model, and user authentication.
        """
        self.master = master
        master.title("Condition Monitoring System for Pressure Vessels")
        master.geometry("1280x720")
        master.resizable(False, False)

        # --- Apply a custom theme for a modern look ---
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Define custom colors for a dark theme
        self.bg_dark = "#2C2F33"
        self.bg_medium = "#23272A"
        self.text_light = "#FFFFFF"
        self.accent_blue = "#7289DA"
        self.accent_green = "#43B581"
        self.accent_red = "#F04747"

        # Configure styles for various Tkinter widgets
        self.style.configure("TFrame", background=self.bg_dark)
        self.style.configure("Navbar.TFrame", background=self.bg_medium)
        self.style.configure("TLabel", background=self.bg_dark,
                             foreground=self.text_light, font=("Arial", 10))
        self.style.configure("Navbar.TLabel", background=self.bg_medium,
                             foreground=self.text_light, font=("Arial", 14, "bold"))
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabelFrame", background=self.bg_dark,
                             foreground=self.text_light)
        self.style.configure("TCheckbutton", background=self.bg_dark,
                             foreground=self.text_light)
        self.style.configure("Horizontal.TScale", background=self.bg_dark)

        # Robot arm variables
        self.theta1 = 90.0
        self.theta2 = 90.0
        self.link1_length = 70
        self.link2_length = 60

        # Sensor data
        self.temperature = tk.StringVar(value="-- °C")
        self.ultrasonic_distance = tk.StringVar(value="-- cm")
        self.pressure = tk.StringVar(value="--")

        # Serial communication
        self.serial_port = "COM3"
        self.baud_rate = 9600
        self.arduino = None
        self.sensor_read_thread = None

        # Camera / recording
        self.camera = None
        self.camera_running = False
        self.camera_thread = None
        self.recording = False
        self.video_writer = None
        self.record_button_text = tk.StringVar(value="Start Recording")

        # YOLOv8
        self.yolov8_model = None
        self.yolov8_model_path = "best.pt"
        self.yolov8_class_names = []
        self.yolov8_colors = []
        self.enable_defect = tk.BooleanVar(value=True)

        # HMI state
        self.pages = {}
        self.current_page = None
        self.logged_in = False

        # Authentication
        # The screenshots show bcrypt.checkpw() against self.users[username].
        # The concrete stored credentials are not visible in the PDF screenshot,
        # so this is a safe runnable default.
        self.users = {
            "admin": bcrypt.hashpw(b"admin", bcrypt.gensalt())
        }

        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()

        self.main_container = None
        self.navbar_frame = None
        self.content_frame = None
        self.status_bar = None
        self.login_frame = None
        self.login_status_label = None

        self.create_login_widgets()

    def create_login_widgets(self):
        """Creates the login and authentication interface."""
        if self.login_frame is not None:
            try:
                self.login_frame.destroy()
            except tk.TclError:
                pass

        self.login_frame = ttk.Frame(self.master, style="TFrame")
        self.login_frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            self.login_frame,
            text="Condition Monitoring System",
            font=("Arial", 24, "bold"),
            background=self.bg_dark,
            foreground=self.text_light
        )
        title.pack(pady=40)

        username_label_frame = ttk.Frame(self.login_frame, style="TFrame")
        username_label_frame.pack(pady=10)

        ttk.Label(
            username_label_frame,
            text="Username:",
            background=self.bg_dark,
            foreground=self.text_light,
            font=("Arial", 12)
        ).pack(side=tk.LEFT, padx=10)

        self.username_entry = ttk.Entry(
            username_label_frame,
            textvariable=self.username_var,
            style="TEntry",
            width=30,
            font=("Arial", 12)
        )
        self.username_entry.pack(side=tk.LEFT, padx=5)

        password_label_frame = ttk.Frame(self.login_frame, style="TFrame")
        password_label_frame.pack(pady=10)

        ttk.Label(
            password_label_frame,
            text="Password:",
            background=self.bg_dark,
            foreground=self.text_light,
            font=("Arial", 12)
        ).pack(side=tk.LEFT, padx=10)

        self.password_entry = ttk.Entry(
            password_label_frame,
            textvariable=self.password_var,
            show="*",
            style="TEntry",
            width=30,
            font=("Arial", 12)
        )
        self.password_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            self.login_frame,
            text="Login",
            command=self.attempt_login
        ).pack(pady=20)

        self.login_status_label = ttk.Label(
            self.login_frame,
            text="",
            background=self.bg_dark,
            foreground=self.accent_red,
            font=("Arial", 10)
        )
        self.login_status_label.pack(pady=5)

        self.master.bind("<Return>", lambda event: self.attempt_login())

    def attempt_login(self):
        """
        Attempts to authenticate the user based on the entered username and password.
        If successful, it transitions to the main HMI interface.
        """
        username = self.username_var.get()
        password = self.password_var.get()

        if username in self.users:
            if bcrypt.checkpw(password.encode('utf-8'), self.users[username]):
                self.logged_in = True
                self.login_frame.destroy()
                self.create_main_hmi_widgets()
                self.status_bar.config(text=f"Logged in as {username}.")
                self.load_yolov8_model()
                self.connect_arduino()
                self.start_camera_feed()
            else:
                self.login_status_label.config(text="Invalid password.")
        else:
            self.login_status_label.config(text="Username not found.")

    def create_main_hmi_widgets(self):
        """
        Creates the main HMI interface, including the navigation sidebar and content area.
        This is called only after successful authentication.
        """
        self.master.config(background=self.bg_dark)

        self.main_container = ttk.Frame(
            self.master, padding="10", style="TFrame"
        )
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # --- Navigation Sidebar ---
        self.navbar_frame = ttk.Frame(
            self.main_container,
            width=120,
            relief=tk.RAISED,
            borderwidth=2,
            padding="10",
            style="Navbar.TFrame"
        )
        self.navbar_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.navbar_frame.pack_propagate(False)

        ttk.Label(
            self.navbar_frame,
            text="Menu",
            style="Navbar.TLabel"
        ).pack(pady=10)

        # Navigation Buttons
        ttk.Button(
            self.navbar_frame,
            text="Home",
            command=lambda: self.show_page("home")
        ).pack(pady=5, fill=tk.X)

        ttk.Button(
            self.navbar_frame,
            text="Manual",
            command=lambda: self.show_page("manual_control")
        ).pack(pady=5, fill=tk.X)

        ttk.Button(
            self.navbar_frame,
            text="Autonomous",
            command=lambda: self.show_page("autonomous_control")
        ).pack(pady=5, fill=tk.X)

        # Logout Button at the bottom of the sidebar
        ttk.Button(
            self.navbar_frame,
            text="Logout",
            command=self.logout
        ).pack(pady=5, fill=tk.X, side=tk.BOTTOM)

        # --- Content Area ---
        self.content_frame = ttk.Frame(
            self.main_container,
            relief=tk.SUNKEN,
            borderwidth=2,
            padding="10",
            style="TFrame"
        )
        self.content_frame.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # --- Add a status bar at the bottom of the main window ---
        self.status_bar = ttk.Label(
            self.master,
            text="",
            relief=tk.SUNKEN,
            anchor=tk.W,
            background=self.bg_medium,
            foreground=self.text_light,
            font=("Arial", 9)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.create_pages()
        self.show_page("home")
        self.update_robot_arm_display()

    def logout(self):
        """
        Logs out the current user, stops ongoing processes (camera, serial),
        and returns to the login screen.
        """
        self.logged_in = False

        # Stop camera feed if running
        if self.camera_running:
            self.camera_running = False
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=1)

        # Stop recording if active
        if self.recording:
            self.stop_recording()

        # Close Arduino serial connection if open
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino serial connection closed.")

        # Destroy all widgets from the main HMI interface
        if self.main_container:
            self.main_container.destroy()
            self.main_container = None

        if self.status_bar:
            self.status_bar.destroy()
            self.status_bar = None

        # Clear username/password fields and show the login screen again
        self.username_var.set("")
        self.password_var.set("")
        self.create_login_widgets()
        self.login_status_label.config(text="Logged out successfully.")

    def show_page(self, page_name):
        """Shows the selected HMI page."""
        if not self.pages:
            return

        for page in self.pages.values():
            page.pack_forget()

        page = self.pages.get(page_name)
        if page:
            page.pack(fill=tk.BOTH, expand=True)
            self.current_page = page_name

    def create_pages(self):
        """
        Creates and stores all the different content pages (frames) for the HMI.
        """
        # --- Home Page ---
        home_page = ttk.Frame(
            self.content_frame, padding="10", style="TFrame"
        )
        self.pages["home"] = home_page

        ttk.Label(
            home_page,
            text="Welcome to HMI",
            font=("Arial", 24, "bold"),
            background=self.bg_dark,
            foreground=self.text_light
        ).pack(pady=50)

        ttk.Label(
            home_page,
            text="This is your condition monitoring system for pressure vessels.",
            font=("Arial", 14),
            background=self.bg_dark,
            foreground=self.text_light
        ).pack(pady=10)

        ttk.Label(
            home_page,
            text="Use the navigation on the left to switch between control modes and view sensor data.",
            font=("Arial", 12),
            background=self.bg_dark,
            foreground=self.text_light
        ).pack(pady=10)

        # --- Live Sensor Readings (Moved to Home Page) ---
        ttk.Label(
            home_page,
            text="Live Sensor Readings",
            font=("Arial", 14, "bold"),
            background=self.bg_dark,
            foreground=self.text_light
        ).pack(pady=10)

        self.sensor_frame = ttk.LabelFrame(
            home_page,
            text="Live Sensor Readings",
            padding="5",
            style="TLabelFrame"
        )
        self.sensor_frame.pack(fill=tk.X, pady=5, padx=50)

        self.sensor_frame.grid_columnconfigure(0, weight=1)
        self.sensor_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(
            self.sensor_frame,
            text="Temperature:",
            font=("Arial", 10, "bold"),
            background=self.bg_dark,
            foreground=self.text_light
        ).grid(row=0, column=0, padx=10, pady=5, sticky="e")

        ttk.Label(
            self.sensor_frame,
            textvariable=self.temperature,
            font=("Arial", 10),
            background=self.bg_dark,
            foreground=self.accent_green
        ).grid(row=0, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(
            self.sensor_frame,
            text="Ultrasonic Distance:",
            font=("Arial", 10, "bold"),
            background=self.bg_dark,
            foreground=self.text_light
        ).grid(row=1, column=0, padx=10, pady=5, sticky="e")

        ttk.Label(
            self.sensor_frame,
            textvariable=self.ultrasonic_distance,
            font=("Arial", 10),
            background=self.bg_dark,
            foreground=self.accent_green
        ).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(
            self.sensor_frame,
            text="Pressure (Simulated):",
            font=("Arial", 10, "bold"),
            background=self.bg_dark,
            foreground=self.text_light
        ).grid(row=2, column=0, padx=10, pady=5, sticky="e")

        ttk.Label(
            self.sensor_frame,
            textvariable=self.pressure,
            font=("Arial", 10),
            background=self.bg_dark,
            foreground=self.accent_green
        ).grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # --- Manual Control Page ---
        manual_control_page = ttk.Frame(
            self.content_frame, padding="10", style="TFrame"
        )
        self.pages["manual_control"] = manual_control_page

        # Divide the manual control page into left and right panels
        self.left_panel_manual = ttk.Frame(
            manual_control_page,
            relief=tk.RAISED,
            borderwidth=2,
            padding="10",
            style="TFrame"
        )
        self.left_panel_manual.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        self.right_panel_manual = ttk.Frame(
            manual_control_page,
            relief=tk.RAISED,
            borderwidth=2,
            padding="10",
            style="TFrame"
        )
        self.right_panel_manual.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # Robot Arm Control & Visualization (left panel)
        ttk.Label(
            self.left_panel_manual,
            text="Camera Positioning and Control",
            font=("Arial", 14, "bold"),
            background=self.bg_dark,
            foreground=self.text_light
        ).pack(pady=5)

        ttk.Label(
            self.left_panel_manual,
            text="Click on the grid to set a target point for the arm.",
            font=("Arial", 9),
            background=self.bg_dark,
            foreground=self.text_light
        ).pack(pady=2)

        self.canvas = tk.Canvas(
            self.left_panel_manual,
            width=250,
            height=180,
            bg="#36393F",
            borderwidth=2,
            relief=tk.SUNKEN
        )
        self.canvas.pack(pady=5, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.angle_control_frame = ttk.LabelFrame(
            self.left_panel_manual,
            text="Manual Angle Control",
            padding="5",
            style="TLabelFrame"
        )
        self.angle_control_frame.pack(fill=tk.X, pady=5)

        # Sliders for manual angle control
        self.theta1_label = ttk.Label(
            self.angle_control_frame,
            text=f"Base Angle (Theta1): {self.theta1:.1f}°",
            background=self.bg_dark,
            foreground=self.text_light,
            font=("Arial", 9)
        )
        self.theta1_label.pack(pady=2)

        self.theta1_slider = ttk.Scale(
            self.angle_control_frame,
            from_=0,
            to=180,
            orient=tk.HORIZONTAL,
            command=self.update_theta1_from_slider,
            length=150,
            style="Horizontal.TScale"
        )
        self.theta1_slider.set(self.theta1)
        self.theta1_slider.pack(fill=tk.X, expand=True)

        self.theta2_label = ttk.Label(
            self.angle_control_frame,
            text=f"Elbow Angle (Theta2): {self.theta2:.1f}°",
            background=self.bg_dark,
            foreground=self.text_light,
            font=("Arial", 9)
        )
        self.theta2_label.pack(pady=2)

        self.theta2_slider = ttk.Scale(
            self.angle_control_frame,
            from_=0,
            to=180,
            orient=tk.HORIZONTAL,
            command=self.update_theta2_from_slider,
            length=150,
            style="Horizontal.TScale"
        )
        self.theta2_slider.set(self.theta2)
        self.theta2_slider.pack(fill=tk.X, expand=True)

        # --- Chassis Control (Moved to left_panel_manual) ---
        self.chassis_control_frame = ttk.LabelFrame(
            self.left_panel_manual,
            text="Chassis Control",
            padding="5",
            style="TLabelFrame"
        )
        self.chassis_control_frame.pack(fill=tk.X, pady=5)

        chassis_button_frame = ttk.Frame(
            self.chassis_control_frame,
            style="TFrame"
        )
        chassis_button_frame.pack(pady=2)

        ttk.Button(
            chassis_button_frame,
            text="⬆",
            command=self.move_forward
        ).grid(row=0, column=1, padx=2, pady=2)

        ttk.Button(
            chassis_button_frame,
            text="⬅",
            command=self.turn_left
        ).grid(row=1, column=0, padx=2, pady=2)

        ttk.Button(
            chassis_button_frame,
            text="⏹",
            command=self.chassis_stop
        ).grid(row=1, column=1, padx=2, pady=2)

        ttk.Button(
            chassis_button_frame,
            text="➡",
            command=self.turn_right
        ).grid(row=1, column=2, padx=2, pady=2)

        ttk.Button(
            chassis_button_frame,
            text="⬇",
            command=self.move_backward
        ).grid(row=2, column=1, padx=2, pady=2)

        # --- Manipulator Controls ---
        self.create_manipulator_controls()

        # --- Camera Live Feed ---
        self.camera_frame = ttk.LabelFrame(
            self.right_panel_manual,
            text="Camera Live Feed (Cracks Detection)",
            padding="5",
            style="TLabelFrame"
        )
        self.camera_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.camera_label = ttk.Label(
            self.camera_frame,
            text="Camera Feed Placeholder\n(Requires OpenCV and YOLOv8)",
            background="#454545",
            foreground="white",
            font=("Arial", 10),
            justify=tk.CENTER
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # Video recording button
        self.record_button = ttk.Button(
            self.right_panel_manual,
            textvariable=self.record_button_text,
            command=self.toggle_recording
        )
        self.record_button.pack(pady=5, fill=tk.X)

        # --- Autonomous Control Page ---
        autonomous_page = ttk.Frame(
            self.content_frame,
            padding="10",
            style="TFrame"
        )
        self.pages["autonomous_control"] = autonomous_page

        ttk.Label(
            autonomous_page,
            text="Autonomous Control",
            font=("Arial", 20, "bold"),
            background=self.bg_dark,
            foreground=self.text_light
        ).pack(pady=20)

        ttk.Label(
            autonomous_page,
            text="Autonomous inspection control",
            font=("Arial", 12),
            background=self.bg_dark,
            foreground=self.text_light
        ).pack(pady=10)

        ttk.Checkbutton(
            autonomous_page,
            text="Enable Defect Detection",
            variable=self.enable_defect
        ).pack(pady=10)

        ttk.Button(
            autonomous_page,
            text="Start Camera Feed",
            command=self.start_camera_feed
        ).pack(pady=5)

        ttk.Button(
            autonomous_page,
            text="Stop Camera Feed",
            command=self.stop_camera_feed
        ).pack(pady=5)

    def create_manipulator_controls(self):
        """Creates the manipulator 1, 2 and 3 controls shown in the HMI screenshots."""

        # --- MANIPULATOR 1 Linear Movement Control (1/3) ---
        self.manipulator1_linear_frame = ttk.LabelFrame(
            self.left_panel_manual,
            text="Manipulator 1 (Linear)",
            padding="5",
            style="TLabelFrame"
        )
        self.manipulator1_linear_frame.pack(fill=tk.X, pady=5)

        m1l_button_frame = ttk.Frame(
            self.manipulator1_linear_frame,
            style="TFrame"
        )
        m1l_button_frame.pack(fill=tk.X, pady=2)

        ttk.Button(
            m1l_button_frame,
            text="Extract",
            command=self.manipulator1_extract
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        ttk.Button(
            m1l_button_frame,
            text="Retract",
            command=self.manipulator1_retract
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        ttk.Button(
            m1l_button_frame,
            text="⏹ Stop",
            command=self.manipulator1_stop
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        # --- MANIPULATOR 1 Angular Movement Control (Gripper) ---
        self.manipulator1_angular_frame = ttk.LabelFrame(
            self.left_panel_manual,
            text="Manipulator 1 (Angular Gripper)",
            padding="5",
            style="TLabelFrame"
        )
        self.manipulator1_angular_frame.pack(fill=tk.X, pady=5)

        m1a_button_frame = ttk.Frame(
            self.manipulator1_angular_frame,
            style="TFrame"
        )
        m1a_button_frame.pack(fill=tk.X, pady=2)

        ttk.Button(
            m1a_button_frame,
            text="Open",
            command=self.manipulator_full_open
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        ttk.Button(
            m1a_button_frame,
            text="Close",
            command=self.manipulator_full_close
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        ttk.Button(
            m1a_button_frame,
            text="⏹ Stop",
            command=self.manipulator_stop
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        # --- MANIPULATOR 2 Linear Movement Control (2/3) ---
        self.manipulator2_linear_frame = ttk.LabelFrame(
            self.left_panel_manual,
            text="Manipulator 2 (Linear)",
            padding="5",
            style="TLabelFrame"
        )
        self.manipulator2_linear_frame.pack(fill=tk.X, pady=5)

        m2l_button_frame = ttk.Frame(
            self.manipulator2_linear_frame,
            style="TFrame"
        )
        m2l_button_frame.pack(fill=tk.X, pady=2)

        ttk.Button(
            m2l_button_frame,
            text="Extract",
            command=self.manipulator2_extract
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        ttk.Button(
            m2l_button_frame,
            text="Retract",
            command=self.manipulator2_retract
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        ttk.Button(
            m2l_button_frame,
            text="⏹ Stop",
            command=self.manipulator2_stop
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        # --- MANIPULATOR 3 Linear Movement Control (3/3) ---
        self.manipulator3_linear_frame = ttk.LabelFrame(
            self.left_panel_manual,
            text="Manipulator 3 (Linear)",
            padding="5",
            style="TLabelFrame"
        )
        self.manipulator3_linear_frame.pack(fill=tk.X, pady=5)

        m3l_button_frame = ttk.Frame(
            self.manipulator3_linear_frame,
            style="TFrame"
        )
        m3l_button_frame.pack(fill=tk.X, pady=2)

        ttk.Button(
            m3l_button_frame,
            text="Extract",
            command=self.manipulator3_extract
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        ttk.Button(
            m3l_button_frame,
            text="Retract",
            command=self.manipulator3_retract
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

        ttk.Button(
            m3l_button_frame,
            text="⏹ Stop",
            command=self.manipulator3_stop
        ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)

    # --- Slider callbacks ---
    def update_theta1_from_slider(self, value):
        """
        Updates the base angle (theta1) based on the slider value and redraws the arm.
        Sends the new angles to the Arduino.
        """
        self.theta1 = float(value)
        self.theta1_label.config(
            text=f"Base Angle (Theta1): {self.theta1:.1f}°"
        )
        self.update_robot_arm_display()
        self.send_angles_to_arduino(self.theta1, self.theta2)

    def update_theta2_from_slider(self, value):
        """
        Updates the elbow angle (theta2) based on the slider value and redraws the arm.
        Sends the new angles to the Arduino.
        """
        self.theta2 = float(value)
        self.theta2_label.config(
            text=f"Elbow Angle (Theta2): {self.theta2:.1f}°"
        )
        self.update_robot_arm_display()
        self.send_angles_to_arduino(self.theta1, self.theta2)

    def update_robot_arm_display(self):
        """
        Redraws the robot arm on the canvas based on the current angles.
        Also draws the grid, target point (if set), and current end-effector coordinates.
        """
        if self.current_page == "manual_control" and hasattr(self, "canvas"):
            self.canvas.delete("all")

            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width <= 1:
                canvas_width = 250
            if canvas_height <= 1:
                canvas_height = 180

            # Draw a simple coordinate grid
            for x in range(0, int(canvas_width), 25):
                self.canvas.create_line(
                    x, 0, x, canvas_height,
                    fill="#4A4A4A"
                )
            for y in range(0, int(canvas_height), 25):
                self.canvas.create_line(
                    0, y, canvas_width, y,
                    fill="#4A4A4A"
                )

            base_x = canvas_width / 2
            base_y = canvas_height - 20

            theta1_rad = math.radians(self.theta1 - 90)
            theta2_rad = math.radians(self.theta2 - 90)

            joint1_x = base_x + self.link1_length * math.cos(theta1_rad)
            joint1_y = base_y - self.link1_length * math.sin(theta1_rad)

            joint2_angle = theta1_rad + theta2_rad
            end_x = joint1_x + self.link2_length * math.cos(joint2_angle)
            end_y = joint1_y - self.link2_length * math.sin(joint2_angle)

            # Robot links
            self.canvas.create_line(
                base_x, base_y, joint1_x, joint1_y,
                fill="#7289DA", width=6
            )
            self.canvas.create_line(
                joint1_x, joint1_y, end_x, end_y,
                fill="#43B581", width=6
            )

            # Joints
            self.canvas.create_oval(
                base_x - 6, base_y - 6,
                base_x + 6, base_y + 6,
                fill="#FFFFFF", outline=""
            )
            self.canvas.create_oval(
                joint1_x - 6, joint1_y - 6,
                joint1_x + 6, joint1_y + 6,
                fill="#FFFFFF", outline=""
            )
            self.canvas.create_oval(
                end_x - 5, end_y - 5,
                end_x + 5, end_y + 5,
                fill="#F04747", outline=""
            )

            if hasattr(self, "target_point") and self.target_point:
                tx, ty = self.target_point
                self.canvas.create_oval(
                    tx - 5, ty - 5, tx + 5, ty + 5,
                    outline="#F04747", width=2
                )

    def on_canvas_click(self, event):
        """Calculates inverse kinematics from a clicked canvas target."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        base_x = canvas_width / 2
        base_y = canvas_height - 20

        x = event.x - base_x
        y = base_y - event.y

        result = self.calculate_inverse_kinematics(x, y)

        if result is None:
            if self.status_bar:
                self.status_bar.config(
                    text="Target point is outside the reachable workspace.",
                    foreground=self.accent_red
                )
            return

        self.target_point = (event.x, event.y)

        self.theta1 = max(0, min(180, result["theta1"]))
        self.theta2 = max(0, min(180, result["theta2"]))

        self.theta1_slider.set(self.theta1)
        self.theta2_slider.set(self.theta2)

        self.theta1_label.config(
            text=f"Base Angle (Theta1): {self.theta1:.1f}°"
        )
        self.theta2_label.config(
            text=f"Elbow Angle (Theta2): {self.theta2:.1f}°"
        )

        self.update_robot_arm_display()
        self.send_angles_to_arduino(self.theta1, self.theta2)

    def calculate_inverse_kinematics(self, x, y):
        L1 = self.link1_length
        L2 = self.link2_length

        D_squared = x**2 + y**2
        D = math.sqrt(D_squared)

        if D > (L1 + L2) or D < abs(L1 - L2):
            return None

        cos_alpha2 = (D_squared - L1**2 - L2**2) / (2 * L1 * L2)
        cos_alpha2 = max(-1, min(1, cos_alpha2))
        alpha2 = math.acos(cos_alpha2)

        new_theta2_rad = alpha2

        beta = math.atan2(y, x)

        cos_gamma = (L1**2 + D_squared - L2**2) / (2 * L1 * D)
        cos_gamma = max(-1, min(1, cos_gamma))
        gamma = math.acos(cos_gamma)

        new_theta1_rad = beta - gamma

        deg_theta1 = math.degrees(new_theta1_rad)
        deg_theta2 = math.degrees(new_theta2_rad)

        return {"theta1": deg_theta1, "theta2": deg_theta2}

    def send_angles_to_arduino(self, t1, t2):
        """
        Sends the calculated angles (t1, t2) to the Arduino via serial communication.
        The data format is "T1:angle1,T2:angle2\n".
        """
        if self.arduino and self.arduino.is_open:
            try:
                data_to_send = f"T1:{int(t1)},T2:{int(t2)}\n"
                print(f"[Python HMI] Sending to Arduino: '{data_to_send.strip()}'")
                self.arduino.write(data_to_send.encode())
                if self.status_bar:
                    self.status_bar.config(
                        text=f"Sent to Arduino: {data_to_send.strip()}"
                    )
            except serial.SerialException as e:
                print(f"Serial write error: {e}")
                if self.status_bar:
                    self.status_bar.config(
                        text=f"Serial Write Error: {e}",
                        foreground=self.accent_red
                    )
            except Exception as e:
                print(f"Error sending data to Arduino: {e}")
                if self.status_bar:
                    self.status_bar.config(
                        text=f"Error sending to Arduino: {e}",
                        foreground=self.accent_red
                    )
        else:
            if self.status_bar:
                self.status_bar.config(
                    text="Arduino not connected.",
                    foreground=self.accent_red
                )

    # --- Chassis Control Callbacks ---
    def move_forward(self):
        self.send_command_to_arduino("C:F\n")  # Chassis: Forward
        if self.status_bar:
            self.status_bar.config(text="Chassis: Moving Forward")

    def turn_left(self):
        self.send_command_to_arduino("C:L\n")  # Chassis: Left
        if self.status_bar:
            self.status_bar.config(text="Chassis: Turning Left")

    def turn_right(self):
        self.send_command_to_arduino("C:R\n")  # Chassis: Right
        if self.status_bar:
            self.status_bar.config(text="Chassis: Turning Right")

    def move_backward(self):
        self.send_command_to_arduino("C:B\n")  # Chassis: Backward
        if self.status_bar:
            self.status_bar.config(text="Chassis: Moving Backward")

    def chassis_stop(self):
        self.send_command_to_arduino("C:S\n")  # Chassis: Stop
        if self.status_bar:
            self.status_bar.config(text="Chassis: Stopped")

    # --- Manipulator 1 Linear Movement Control Callbacks ---
    def manipulator1_extract(self):
        self.send_command_to_arduino("M1:E\n")
        if self.status_bar:
            self.status_bar.config(text="Manipulator 1: Extracting")

    def manipulator1_retract(self):
        self.send_command_to_arduino("M1:R\n")
        if self.status_bar:
            self.status_bar.config(text="Manipulator 1: Retracting")

    def manipulator1_stop(self):
        self.send_command_to_arduino("M1:S\n")
        if self.status_bar:
            self.status_bar.config(text="Manipulator 1: Stopped")

    # --- Manipulator Angular Movement Control Callbacks ---
    def manipulator_full_open(self):
        self.send_command_to_arduino("MA:O\n")  # Manipulator Angular: Open
        if self.status_bar:
            self.status_bar.config(text="Manipulator: Opening Gripper")

    def manipulator_full_close(self):
        self.send_command_to_arduino("MA:C\n")  # Manipulator Angular: Close
        if self.status_bar:
            self.status_bar.config(text="Manipulator: Closing Gripper")

    def manipulator_stop(self):
        self.send_command_to_arduino("MA:S\n")
        if self.status_bar:
            self.status_bar.config(text="Manipulator: Stopped")

    # --- Manipulator 2 Linear Movement Control Callbacks ---
    def manipulator2_extract(self):
        self.send_command_to_arduino("M2:E\n")
        if self.status_bar:
            self.status_bar.config(text="Manipulator 2: Extracting")

    def manipulator2_retract(self):
        self.send_command_to_arduino("M2:R\n")
        if self.status_bar:
            self.status_bar.config(text="Manipulator 2: Retracting")

    def manipulator2_stop(self):
        self.send_command_to_arduino("M2:S\n")
        if self.status_bar:
            self.status_bar.config(text="Manipulator 2: Stopped")

    # --- Manipulator 3 Linear Movement Control Callbacks ---
    def manipulator3_extract(self):
        self.send_command_to_arduino("M3:E\n")
        if self.status_bar:
            self.status_bar.config(text="Manipulator 3: Extracting")

    def manipulator3_retract(self):
        self.send_command_to_arduino("M3:R\n")
        if self.status_bar:
            self.status_bar.config(text="Manipulator 3: Retracting")

    def manipulator3_stop(self):
        self.send_command_to_arduino("M3:S\n")
        if self.status_bar:
            self.status_bar.config(text="Manipulator 3: Stopped")

    def send_command_to_arduino(self, command):
        """Generic function to send commands to Arduino."""
        if self.arduino and self.arduino.is_open:
            try:
                print(f"[Python HMI] Sending command to Arduino: '{command.strip()}'")
                self.arduino.write(command.encode())
            except serial.SerialException as e:
                print(f"Serial write error: {e}")
                if self.status_bar:
                    self.status_bar.config(
                        text=f"Serial Write Error: {e}",
                        foreground=self.accent_red
                    )
            except Exception as e:
                print(f"Error sending command to Arduino: {e}")
                if self.status_bar:
                    self.status_bar.config(
                        text=f"Error sending command to Arduino: {e}",
                        foreground=self.accent_red
                    )
        else:
            if self.status_bar:
                self.status_bar.config(
                    text="Arduino not connected.",
                    foreground=self.accent_red
                )

    # --- Serial connection and data management ---
    def connect_arduino(self):
        """
        Attempts to establish a serial connection with the Arduino.
        """
        print(f"[Arduino] Attempting to connect to {self.serial_port} at {self.baud_rate} baud...")

        if self.status_bar:
            self.status_bar.config(
                text=f"Connecting to Arduino on {self.serial_port}...",
                foreground=self.text_light
            )

        try:
            self.arduino = serial.Serial(
                self.serial_port,
                self.baud_rate,
                timeout=1
            )
            time.sleep(2)  # Give the Arduino time to reset

            if self.status_bar:
                self.status_bar.config(
                    text=f"Successfully connected to Arduino on {self.serial_port}.",
                    foreground=self.accent_green
                )

            print("Successfully connected to Arduino.")

            # Start a separate thread to read from the serial port
            self.sensor_read_thread = threading.Thread(
                target=self.read_serial_data,
                daemon=True
            )
            self.sensor_read_thread.start()

        except serial.SerialException as e:
            print(f"Could not connect to Arduino: {e}")
            if self.status_bar:
                self.status_bar.config(
                    text=f"Could not connect to Arduino: {e}",
                    foreground=self.accent_red
                )

    def read_serial_data(self):
        """
        Reads data from the Arduino serial port in a separate thread and
        updates the HMI sensor labels.
        """
        while self.arduino and self.arduino.is_open:
            try:
                line = self.arduino.readline().decode('utf-8').strip()
                if line:
                    self.parse_sensor_data(line)
            except serial.SerialException:
                print("Serial port disconnected.")
                if self.status_bar:
                    self.master.after(
                        0,
                        lambda: self.status_bar.config(
                            text="Serial port disconnected.",
                            foreground=self.accent_red
                        )
                    )
                break
            except Exception as e:
                print(f"Error reading serial data: {e}")
                break

    def parse_sensor_data(self, data):
        """
        Parses a string of sensor data received from the Arduino.
        Expected format: "T:temp,D:dist,P:pres"
        """
        try:
            parts = data.split(',')
            for part in parts:
                if part.startswith("T:"):
                    self.master.after(
                        0,
                        lambda value=part[2:]: self.temperature.set(f"{value}°C")
                    )
                elif part.startswith("D:"):
                    self.master.after(
                        0,
                        lambda value=part[2:]: self.ultrasonic_distance.set(f"{value} cm")
                    )
                elif part.startswith("P:"):
                    self.master.after(
                        0,
                        lambda value=part[2:]: self.pressure.set(value)
                    )
        except Exception as e:
            print(f"Error parsing sensor data: {e}")

    # --- Camera / YOLOv8 ---
    def load_yolov8_model(self):
        """
        Loads the YOLOv8 model from the specified path.
        Also, loads class names and generates colors for visualization.
        """
        if torch.cuda.is_available():
            device = 'cuda'
            print("[YOLOv8] CUDA is available. Using GPU for inference.")
        else:
            device = 'cpu'
            print("[YOLOv8] CUDA not available. Using CPU for inference.")

        # --- Dynamically find the path to the 'best.pt' file relative to the script's location ---
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Create the full path to the 'best.pt' file
            model_path = os.path.join(script_dir, self.yolov8_model_path)
        except NameError:
            # Fallback for environments where __file__ is not defined
            script_dir = os.getcwd()
            model_path = os.path.join(script_dir, self.yolov8_model_path)

        try:
            # Load the YOLOv8 model using the dynamically created path
            self.yolov8_model = YOLO(model_path)
            self.yolov8_class_names = self.yolov8_model.names

            # Generate visualization colors
            class_count = len(self.yolov8_class_names)
            self.yolov8_colors = [
                tuple(np.random.randint(0, 255, 3).tolist())
                for _ in range(class_count)
            ]

            print(f"[YOLOv8] Model loaded successfully from {model_path}")
        except Exception as e:
            self.yolov8_model = None
            print(f"[YOLOv8] Error loading model: {e}")
            if self.status_bar:
                self.status_bar.config(
                    text=f"YOLOv8 model error: {e}",
                    foreground=self.accent_red
                )

    def start_camera_feed(self):
        """Starts the OpenCV camera capture and the camera processing thread."""
        if self.camera_running:
            return

        try:
            self.camera = cv2.VideoCapture(CAMERA_SOURCE)

            if not self.camera.isOpened():
                if self.status_bar:
                    self.status_bar.config(
                        text="Error opening camera source.",
                        foreground=self.accent_red
                    )
                return

            self.camera_running = True

            if self.status_bar:
                self.status_bar.config(
                    text="Camera feed started.",
                    foreground=self.accent_green
                )

            self.camera_thread = threading.Thread(
                target=self.update_camera_feed,
                daemon=True
            )
            self.camera_thread.start()

        except Exception as e:
            print(f"Error starting camera: {e}")
            if self.status_bar:
                self.status_bar.config(
                    text=f"Camera error: {e}",
                    foreground=self.accent_red
                )

    def stop_camera_feed(self):
        """Stops the camera feed safely."""
        self.camera_running = False

        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1)

        if self.camera:
            self.camera.release()
            self.camera = None

        if hasattr(self, "camera_label"):
            self.master.after(
                0,
                lambda: self.camera_label.config(
                    image="",
                    text="Camera Feed Stopped"
                )
            )

    def update_camera_feed(self):
        """
        Reads camera frames, optionally performs YOLOv8 inference,
        displays them in the Tkinter HMI, and records them when enabled.
        """
        while self.camera_running and self.camera:
            try:
                ret, frame = self.camera.read()

                if not ret:
                    continue

                # YOLOv8 inference
                if self.enable_defect.get() and self.yolov8_model is not None:
                    try:
                        results = self.yolov8_model(
                            frame,
                            device=0 if torch.cuda.is_available() else "cpu",
                            verbose=False
                        )

                        for result in results:
                            boxes = result.boxes

                            for box in boxes:
                                confidence = float(box.conf[0])

                                if confidence < 0.5:
                                    continue

                                x1, y1, x2, y2 = map(
                                    int,
                                    box.xyxy[0].tolist()
                                )

                                cls_id = int(box.cls[0])
                                if isinstance(self.yolov8_class_names, dict):
                                    class_name = self.yolov8_class_names.get(
                                        cls_id, str(cls_id)
                                    )
                                else:
                                    class_name = str(cls_id)

                                if self.yolov8_colors:
                                    color = self.yolov8_colors[
                                        cls_id % len(self.yolov8_colors)
                                    ]
                                else:
                                    color = (0, 255, 0)

                                cv2.rectangle(
                                    frame,
                                    (x1, y1),
                                    (x2, y2),
                                    color,
                                    2
                                )

                                label = f"{class_name} {confidence:.2f}"

                                cv2.putText(
                                    frame,
                                    label,
                                    (x1, max(y1 - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2
                                )
                    except Exception as e:
                        print(f"YOLO inference error: {e}")

                # Video recording
                if self.recording:
                    self.write_recording_frame(frame)

                # Convert OpenCV BGR frame to RGB for Tkinter/PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                # Resize for the camera panel
                image.thumbnail((500, 400))

                photo = ImageTk.PhotoImage(image=image)

                self.master.after(
                    0,
                    self.update_camera_label,
                    photo
                )

            except Exception as e:
                print(f"Error updating camera feed: {e}")
                break

    def update_camera_label(self, photo):
        """Updates the camera Label on the Tkinter main thread."""
        if hasattr(self, "camera_label") and self.camera_label.winfo_exists():
            self.camera_label.configure(
                image=photo,
                text=""
            )
            self.camera_label.image = photo

    # --- Video recording ---
    def toggle_recording(self):
        """Starts or stops video recording."""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Initializes timestamped AVI recording."""
        if self.camera is None:
            if self.status_bar:
                self.status_bar.config(
                    text="Camera is not available for recording.",
                    foreground=self.accent_red
                )
            return

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(
            script_dir,
            f"video_record_{timestamp}.avi"
        )

        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.camera.get(cv2.CAP_PROP_FPS)

        if fps <= 0 or math.isnan(fps):
            fps = 20.0

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(
            filename,
            fourcc,
            fps,
            (width, height)
        )

        if not self.video_writer.isOpened():
            self.video_writer = None
            if self.status_bar:
                self.status_bar.config(
                    text="Could not start video recording.",
                    foreground=self.accent_red
                )
            return

        self.recording = True
        self.record_button_text.set("Stop Recording")

        if self.status_bar:
            self.status_bar.config(
                text=f"Recording started: {os.path.basename(filename)}"
            )

    def write_recording_frame(self, frame):
        """Writes a processed frame to the active video file."""
        if self.recording and self.video_writer is not None:
            try:
                self.video_writer.write(frame)
            except Exception as e:
                print(f"Recording error: {e}")

    def stop_recording(self):
        """Stops the current video recording."""
        self.recording = False

        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

        self.record_button_text.set("Start Recording")

        if self.status_bar:
            self.status_bar.config(text="Video recording stopped.")

    def on_close(self):
        """Gracefully releases all resources before application exit."""
        try:
            self.logged_in = False
            self.camera_running = False

            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=1)

            if self.camera:
                self.camera.release()
                self.camera = None

            if self.recording:
                self.stop_recording()

            if self.arduino and self.arduino.is_open:
                self.arduino.close()

        except Exception as e:
            print(f"Shutdown error: {e}")

        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CBMHMI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()