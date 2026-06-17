import os
import csv
import time
import ctypes
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

# Import the custom hardware wrappers provided in your environment
from waveforms_ads import WaveFormsADS
from ttl_trigger import TTLTrigger, TTLTriggerConfig, TTLIdleState, TTLPolarity, TTLTriggerMode
from allied_vision_camera import AlliedVisionCamera, CameraConfig, HardwareTriggerConfig, TriggerActivation, TriggerSelector, AcquisitionMode

class CoreInstrumentApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MOT Control Panel")
        self.geometry("1400x900")
        self.state("zoomed") # Open maximized for visual real estate

        # Initialize Hardware Interface Objects
        self.ads = None
        self.ttl_trig = None
        self.camera = None
        
        # State Arrays and Image Matrices
        self.background_image = None
        self.latest_live_frame = None
        self.latest_pulsed_snapshot = None
        self.live_view_active = True
        
        # Canvas Drag-to-ROI Coordinates
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.current_rect_id = None
        
        # Connect to Devices Safeguarded against immediate missing hardware
        self.init_hardware_connections()

        # Build GUI Frame Panes
        self.create_four_panels()
        
        # Start Live View Processing Loop
        self.start_live_acquisition_thread()

    def init_hardware_connections(self):
        """Secure safe handles to the hardware layers."""
        try:
            self.ads = WaveFormsADS()
            self.ttl_trig = TTLTrigger(self.ads)
            # Initialize DIO 0,1,2 as outputs explicitly if required by platform
            self.ads._dwf.FDwfDigitalIOOutputEnableSet(self.ads._hdwf, ctypes.c_int(0x07)) 
        except Exception as e:
            print(f"[Warning] Analog Discovery device could not connect: {e}")
            self.ads = None
            self.ttl_trig = None

        try:
            # Generate baseline configuration for Allied Vision
            cam_cfg = CameraConfig(exposure_time_us=20000, gain=0.0, brightness=0.0)
            self.camera = AlliedVisionCamera(cam_cfg)
            self.camera.open()
        except Exception as e:
            print(f"[Warning] Allied Vision Camera could not connect: {e}")
            self.camera = None

    def create_four_panels(self):
        """Construct a clean 2x2 responsive grid allocation layout."""
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # 1. Top Left: Pulse Configuration and Waveform Preview
        self.p_top_left = ttk.LabelFrame(self, text="Top Left: Pulse Control Sequence Settings")
        self.p_top_left.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.build_top_left_panel()

        # 2. Top Right: Live Camera Matrix Control & Parameters
        self.p_top_right = ttk.LabelFrame(self, text="Top Right: Camera Interface & Live Video")
        self.p_top_right.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.build_top_right_panel()

        # 3. Bottom Left: Oscilloscope Trace (Fluorescence PD3) & Magnet IO Switches
        self.p_bottom_left = ttk.LabelFrame(self, text="Bottom Left: Fluorescence (PD3) Scope & Magnet Control")
        self.p_bottom_left.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.build_bottom_left_panel()

        # 4. Bottom Right: Signal Processing Matrix (Snapshot Subtraction Array)
        self.p_bottom_right = ttk.LabelFrame(self, text="Bottom Right: Data Extraction & Background Profiles")
        self.p_bottom_right.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        self.build_bottom_right_panel()

    # =========================================================================
    # PANEL BUILDERS & LOGIC SECTIONS
    # =========================================================================

    def build_top_left_panel(self):
        """User parameter dashboard specifying sequencing configurations."""
        container = ttk.Frame(self.p_top_left, padding=8)
        container.pack(fill="both", expand=True)

        # Config Variables
        self.var_time_after_pulse = tk.DoubleVar(value=5.0)  # ms
        self.var_time_between_pulses = tk.DoubleVar(value=1.0) # s
        self.var_pd3_window = tk.DoubleVar(value=50.0)       # ms
        self.var_num_pulses = tk.IntVar(value=1)
        self.var_step_length = tk.DoubleVar(value=1.0)       # ms

        # Form Controls layout grid
        lbl_style = {"sticky": "w", "padx": 4, "pady": 2}
        ttk.Label(container, text="Time after pulse to snap (ms):").grid(row=0, column=0, **lbl_style)
        ttk.Entry(container, textvariable=self.var_time_after_pulse, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(container, text="Time between repeat pulses (s):").grid(row=1, column=0, **lbl_style)
        ttk.Entry(container, textvariable=self.var_time_between_pulses, width=10).grid(row=1, column=1, sticky="w")

        ttk.Label(container, text="Fluorescence PD3 Domain (ms):").grid(row=2, column=0, **lbl_style)
        ttk.Entry(container, textvariable=self.var_pd3_window, width=10).grid(row=2, column=1, sticky="w")

        ttk.Label(container, text="Number of Pulses:").grid(row=3, column=0, **lbl_style)
        ttk.Entry(container, textvariable=self.var_num_pulses, width=10).grid(row=3, column=1, sticky="w")

        ttk.Label(container, text="Step Length Resolution (ms):").grid(row=4, column=0, **lbl_style)
        ttk.Entry(container, textvariable=self.var_step_length, width=10).grid(row=4, column=1, sticky="w")

        # Visual Plot Canvas for Matrix Preview Strategy
        ttk.Label(container, text="Intended Signal Trajectory Preview:").grid(row=5, column=0, columnspan=2, sticky="w", pady=(10,2))
        self.sequence_canvas = tk.Canvas(container, height=140, bg="#1e1e1e", highlightthickness=0)
        self.sequence_canvas.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=4)
        
        # Re-draw the visual timing preview trace every time values shift
        for var in [self.var_time_after_pulse, self.var_time_between_pulses, self.var_pd3_window, self.var_step_length]:
            var.trace_add("write", lambda *args: self.render_sequence_preview_graph())
        self.var_num_pulses.trace_add("write", lambda *args: self.render_sequence_preview_graph())
        
        self.render_sequence_preview_graph()

    def render_sequence_preview_graph(self):
        """Draws a multi-channel timing schematic inside the UI without relying on matplotlib."""
        self.sequence_canvas.delete("all")
        w = self.sequence_canvas.winfo_width() if self.sequence_canvas.winfo_width() > 50 else 400
        h = 140
        
        # Draw horizontal channels lines for Magnet, Shutter, and Camera Triggers
        self.sequence_canvas.create_text(45, 25, text="Mag (DIO 0)", fill="#4caf50", font=("Consolas", 9))
        self.sequence_canvas.create_line(90, 25, w-20, 25, fill="#333333", dash=(4,4))
        
        self.sequence_canvas.create_text(45, 65, text="Shut (DIO 1)", fill="#2196f3", font=("Consolas", 9))
        self.sequence_canvas.create_line(90, 65, w-20, 65, fill="#333333", dash=(4,4))
        
        self.sequence_canvas.create_text(45, 105, text="Cam (DIO 2)", fill="#ff9800", font=("Consolas", 9))
        self.sequence_canvas.create_line(90, 105, w-20, 105, fill="#333333", dash=(4,4))

        # Basic relative geometric pulses step tracking
        try:
            snap_delay = self.var_time_after_pulse.get()
            pd3_domain = self.var_pd3_window.get()
            
            # Map pulse widths graphically onto the localized domain
            start_x = 120
            shutter_end_x = start_x + int(pd3_domain * 2)
            cam_trigger_x = start_x + int(snap_delay * 2)

            # Cap boundaries to canvas width
            shutter_end_x = min(shutter_end_x, w - 20)
            cam_trigger_x = min(cam_trigger_x, w - 20)

            # Draw representative traces
            # Magnet Power Line
            self.sequence_canvas.create_line(90, 25, start_x, 25, fill="#4caf50", width=2)
            self.sequence_canvas.create_line(start_x, 25, start_x, 10, fill="#4caf50", width=2)
            self.sequence_canvas.create_line(start_x, 10, shutter_end_x, 10, fill="#4caf50", width=2)
            self.sequence_canvas.create_line(shutter_end_x, 10, shutter_end_x, 25, fill="#4caf50", width=2)
            self.sequence_canvas.create_line(shutter_end_x, 25, w-20, 25, fill="#4caf50", width=2)

            # Shutter Trace
            self.sequence_canvas.create_line(90, 65, start_x, 65, fill="#2196f3", width=2)
            self.sequence_canvas.create_line(start_x, 65, start_x, 50, fill="#2196f3", width=2)
            self.sequence_canvas.create_line(start_x, 50, shutter_end_x, 50, fill="#2196f3", width=2)
            self.sequence_canvas.create_line(shutter_end_x, 50, shutter_end_x, 65, fill="#2196f3", width=2)
            self.sequence_canvas.create_line(shutter_end_x, 65, w-20, 65, fill="#2196f3", width=2)

            # Camera Sync Trace
            self.sequence_canvas.create_line(90, 105, cam_trigger_x, 105, fill="#ff9800", width=2)
            self.sequence_canvas.create_line(cam_trigger_x, 105, cam_trigger_x, 90, fill="#ff9800", width=2)
            self.sequence_canvas.create_line(cam_trigger_x, 90, cam_trigger_x + 15, 90, fill="#ff9800", width=2)
            self.sequence_canvas.create_line(cam_trigger_x + 15, 90, cam_trigger_x + 15, 105, fill="#ff9800", width=2)
            self.sequence_canvas.create_line(cam_trigger_x + 15, 105, w-20, 105, fill="#ff9800", width=2)
        except Exception:
            pass # Suppress entry parsing hiccups during typing transitions

    def build_top_right_panel(self):
        """Live video viewport and camera configuration settings controls."""
        # FIX: Removed the invalid padding option from the PanedWindow constructor
        main_layout = ttk.PanedWindow(self.p_top_right, orient="horizontal")
        main_layout.pack(fill="both", expand=True)

        # Separate sub-frames for configurations and live video canvas feed
        controls_frame = ttk.Frame(main_layout, padding=4)
        main_layout.add(controls_frame, weight=1)

        video_frame = ttk.Frame(main_layout, padding=4)
        main_layout.add(video_frame, weight=3)

        # Region Of Interest entries setup
        self.roi_x = tk.IntVar(value=0)
        self.roi_y = tk.IntVar(value=0)
        self.roi_w = tk.IntVar(value=1920)
        self.roi_h = tk.IntVar(value=1080)

        roi_box = ttk.LabelFrame(controls_frame, text="Region of Interest Configuration")
        roi_box.pack(fill="x", pady=4, padx=2)

        grid_params = {"sticky": "w", "padx": 2, "pady": 1}
        ttk.Label(roi_box, text="Offset X:").grid(row=0, column=0, **grid_params)
        ttk.Entry(roi_box, textvariable=self.roi_x, width=6).grid(row=0, column=1)
        ttk.Label(roi_box, text="Offset Y:").grid(row=1, column=0, **grid_params)
        ttk.Entry(roi_box, textvariable=self.roi_y, width=6).grid(row=1, column=1)
        ttk.Label(roi_box, text="Width:").grid(row=2, column=0, **grid_params)
        ttk.Entry(roi_box, textvariable=self.roi_w, width=6).grid(row=2, column=1)
        ttk.Label(roi_box, text="Height:").grid(row=3, column=0, **grid_params)
        ttk.Entry(roi_box, textvariable=self.roi_h, width=6).grid(row=3, column=1)
        
        ttk.Button(roi_box, text="Apply Box Settings", command=self.apply_manual_roi_parameters).grid(row=4, column=0, columnspan=2, pady=4)

        # Analog/Digital parameters modifiers
        self.var_exposure = tk.DoubleVar(value=20000.0) # us
        self.var_gain = tk.DoubleVar(value=0.0)
        self.var_brightness = tk.DoubleVar(value=0.0)

        param_box = ttk.LabelFrame(controls_frame, text="Gain & Intensity Settings")
        param_box.pack(fill="x", pady=4, padx=2)

        ttk.Label(param_box, text="Exposure (µs):").grid(row=0, column=0, **grid_params)
        ttk.Entry(param_box, textvariable=self.var_exposure, width=8).grid(row=0, column=1)
        ttk.Label(param_box, text="Gain (dB):").grid(row=1, column=0, **grid_params)
        ttk.Entry(param_box, textvariable=self.var_gain, width=8).grid(row=1, column=1)
        ttk.Label(param_box, text="Black Lvl / Bright:").grid(row=2, column=0, **grid_params)
        ttk.Entry(param_box, textvariable=self.var_brightness, width=8).grid(row=2, column=1)
        
        ttk.Button(param_box, text="Commit Attributes", command=self.apply_camera_attributes).grid(row=3, column=0, columnspan=2, pady=4)

        # Operational Control Buttons Pack
        action_box = ttk.LabelFrame(controls_frame, text="Direct Trigger Operations")
        action_box.pack(fill="x", pady=4, padx=2)

        ttk.Button(action_box, text="Capture Snapshot Now", command=self.execute_immediate_snapshot).pack(fill="x", pady=2)
        ttk.Button(action_box, text="Extract & Save Background", command=self.capture_background_profile).pack(fill="x", pady=2)
        
        self.btn_master_pulse = tk.Button(action_box, text="FIRE MASTER PULSE", bg="#d32f2f", fg="white", font=("Arial", 11, "bold"), command=self.execute_master_pulse_routine)
        self.btn_master_pulse.pack(fill="x", pady=6)

        # Live Display Canvas Layout
        self.camera_canvas = tk.Canvas(video_frame, bg="#0d0d0d", bd=1, relief="sunken")
        self.camera_canvas.pack(fill="both", expand=True)

        # Bounding box selection canvas hooks
        self.camera_canvas.bind("<ButtonPress-1>", self.on_roi_drag_start)
        self.camera_canvas.bind("<B1-Motion>", self.on_roi_dragging)
        self.camera_canvas.bind("<ButtonRelease-1>", self.on_roi_drag_end)

    def build_bottom_left_panel(self):
        """Controls manual magnet lines and graphs raw data arrays from channel zero."""
        container = ttk.Frame(self.p_bottom_left, padding=6)
        container.pack(fill="both", expand=True)

        # Split Controls and Plot Layout
        controls_sub = ttk.Frame(container)
        controls_sub.pack(side="top", fill="x", pady=2)

        # Magnet toggle switches
        self.var_magnet_state = tk.BooleanVar(value=False)
        self.chk_magnet = ttk.Checkbutton(controls_sub, text="Enable Magnet Power (Static DIO 0)", variable=self.var_magnet_state, command=self.toggle_magnet_static_line)
        self.chk_magnet.grid(row=0, column=0, sticky="w", padx=4)

        self.var_sync_pulse = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_sub, text="Synchronize Line with Pulse Sequence", variable=self.var_sync_pulse).grid(row=0, column=1, sticky="w", padx=10)

        # CSV Logging Parameter Widgets
        self.var_save_csv = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_sub, text="Save 'Fluorescence (PD3)' Trace", variable=self.var_save_csv).grid(row=1, column=0, sticky="w", padx=4, pady=4)

        self.var_csv_path = tk.StringVar(value=os.path.join(os.getcwd(), "fluorescence_output.csv"))
        ttk.Entry(controls_sub, textvariable=self.var_csv_path, width=40).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Button(controls_sub, text="Browse Destination", command=self.browse_csv_destination_file).grid(row=1, column=2, padx=4)

        # WaveForms Oscilloscope Trace Visual Canvas Display Component
        ttk.Label(container, text="Oscilloscope Buffer Display: Channel 0 (Fluorescence Array Data)").pack(anchor="w", pady=(6,0))
        
        # FIX: Placed inside a dedicated canvas master frame with pack parameters configured to scale properly
        self.scope_canvas = tk.Canvas(container, bg="#000000", height=180, highlightthickness=0)
        self.scope_canvas.pack(fill="both", expand=True, pady=4)

    def build_bottom_right_panel(self):
        """Maintains image subtraction data arrays, displaying original and processed streams side by side."""
        container = ttk.Frame(self.p_bottom_right, padding=6)
        container.pack(fill="both", expand=True)

        # Config layout definitions
        self.var_auto_save_pulsed_img = tk.BooleanVar(value=False)
        self.var_pulse_filename_base = tk.StringVar(value="pulsed_frame_capture")

        top_config = ttk.Frame(container)
        top_config.pack(side="top", fill="x", pady=2)

        ttk.Checkbutton(top_config, text="Auto-Save Pulse Image Arrays", variable=self.var_auto_save_pulsed_img).grid(row=0, column=0, padx=4, sticky="w")
        ttk.Label(top_config, text="Base Filename:").grid(row=0, column=1, padx=4, sticky="w")
        ttk.Entry(top_config, textvariable=self.var_pulse_filename_base, width=20).grid(row=0, column=2, padx=4, sticky="w")
        ttk.Button(top_config, text="Manually Save Current Snapshot", command=self.save_current_snapshot_manually).grid(row=0, column=3, padx=10, sticky="w")
        ttk.Button(top_config, text="Clear Subtraction Background", command=self.clear_background_buffer).grid(row=0, column=4, padx=4, sticky="w")

        # Two-channel Display Viewport Sub-frames
        viewport_frame = ttk.Frame(container)
        viewport_frame.pack(fill="both", expand=True, pady=4)
        
        # FIX: Explicit grid weighting configurations ensure the left and right camera channels track equally
        viewport_frame.columnconfigure(0, weight=1)
        viewport_frame.columnconfigure(1, weight=1)
        viewport_frame.rowconfigure(0, weight=1)

        # Left Sub-Frame: Resulting Pulsed Snapshot Visualizer
        left_f = ttk.Frame(viewport_frame)
        left_f.grid(row=0, column=0, sticky="nsew", padx=2)
        ttk.Label(left_f, text="Resulting Processed Pulse Snapshot (Subtracted Line)").pack(anchor="n")
        self.lbl_snapshot_display = ttk.Label(left_f, background="black")
        self.lbl_snapshot_display.pack(fill="both", expand=True, pady=2)

        # Right Sub-Frame: Background Calibration Profile Visualizer
        right_f = ttk.Frame(viewport_frame)
        right_f.grid(row=0, column=1, sticky="nsew", padx=2)
        ttk.Label(right_f, text="Active Background Matrix Reference Image").pack(anchor="n")
        self.lbl_background_display = ttk.Label(right_f, background="black")
        self.lbl_background_display.pack(fill="both", expand=True, pady=2)

    # =========================================================================
    # CORE INTERACTION LOGIC & HARDWARE DRIVERS
    # =========================================================================

    def apply_manual_roi_parameters(self):
        """Pass bounded dimensions onto the Allied Vision engine ensuring safe spatial rounding constraints."""
        if not self.camera:
            return
        
        # Allied Vision sensor arrays standardly require width/height alignments divisible by 2 or 4 
        x = (self.roi_x.get() // 2) * 2
        y = (self.roi_y.get() // 2) * 2
        w = (self.roi_w.get() // 4) * 4
        h = (self.roi_h.get() // 4) * 4

        # Enforce positive scaling metrics to prevent downstream micro-code runtime failures
        w = max(16, w)
        h = max(16, h)

        # Update text input matrices with calibrated normalized boundary coordinates
        self.roi_x.set(x)
        self.roi_y.set(y)
        self.roi_w.set(w)
        self.roi_h.set(h)

        try:
            # Reconfigure the internal hardware region definitions safely
            self.camera._config.roi_offset_x = x
            self.camera._config.roi_offset_y = y
            self.camera._config.roi_width = w
            self.camera._config.roi_height = h
            
            # Restart or flash definitions onto live handle dynamically
            if self.camera._cam:
                self.camera._apply_roi(x, y, w, h)
            print(f"[Camera] Bounded ROI applied successfully: {x}, {y}, {w}, {h}")
        except Exception as err:
            messagebox.showerror("ROI Limit Violation", f"The camera rejected these bounding coordinates: {err}")

    def apply_camera_attributes(self):
        """Commit electronic gain levels and timing windows directly onto camera registers."""
        if not self.camera or not self.camera._cam:
            return
        try:
            self.camera.set_exposure_time(self.var_exposure.get())
            self.camera.set_gain(self.var_gain.get())
            self.camera.set_brightness(self.var_brightness.get())
            print("[Camera] Electronic exposure, gain, and offset properties updated.")
        except Exception as e:
            messagebox.showerror("Hardware Communication Error", f"Unable to update camera settings block: {e}")

    def toggle_magnet_static_line(self):
        """Drives manual state level adjustments across the static digital registers."""
        if not self.ads:
            return
        # Ensure pattern generator channels are fully closed to avoid system state locking
        try:
            self.ads._dwf.FDwfDigitalOutReset(self.ads._hdwf)
        except Exception:
            pass
            
        state_bit = 1 if self.var_magnet_state.get() else 0
        try:
            self.ads.digital_io_write_pin(pin=0, value=state_bit)
        except Exception as e:
            print(f"[Error] Failed static register write manipulation: {e}")

    def browse_csv_destination_file(self):
        """Launches localized system directory browser to identify data dump locations."""
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Tables", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.var_csv_path.set(file_path)

    # =========================================================================
    # LIVE VIEW STREAM PROCESSING LOOP
    # =========================================================================

    def start_live_acquisition_thread(self):
        """Asynchronously streams matrices from the camera to avoid blocking the main Tkinter thread loop."""
        def capture_stream_worker():
            while True:
                if self.live_view_active and self.camera and self.camera._cam:
                    try:
                        # Capture a single frame using the software snapshot method
                        frame = self.camera.take_snapshot()
                        if frame is not None:
                            self.latest_live_frame = frame.copy()
                            self.process_and_update_live_canvas(frame)
                    except Exception:
                        pass
                time.sleep(0.033) # Keep updates smooth around ~30 FPS

        threading.Thread(target=capture_stream_worker, daemon=True).start()

    def process_and_update_live_canvas(self, raw_img_matrix):
        """Applies mathematical subtraction steps live before drawing raw frames onto the canvas."""
        processed_frame = raw_img_matrix.copy()
        
        # Execute real-time background absolute value frame subtraction if valid matrices exist
        if self.background_image is not None and self.background_image.shape == processed_frame.shape:
            processed_frame = cv2.absdiff(processed_frame, self.background_image)

        # Scale down large raw array arrays to fit display panels safely
        canvas_w = self.camera_canvas.winfo_width()
        canvas_h = self.camera_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 640, 480

        img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB) if len(processed_frame.shape) == 3 else cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        # Thread-safe interface injection back onto the Tkinter canvas pipeline
        self.after(0, self._render_tk_image_to_canvas, tk_img)

    def _render_tk_image_to_canvas(self, tk_img):
        self._live_tk_image_holder = tk_img # Maintain pointer memory reference to prevent sudden garbage collection drops
        self.camera_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        # Keep bounding box overlay visible on top of the image stream
        if self.current_rect_id:
            self.camera_canvas.tag_raise(self.current_rect_id)

    # =========================================================================
    # DRAG AND DROP BOUNDING BOX (ROI) MAPPING RULES
    # =========================================================================

    def on_roi_drag_start(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        if self.current_rect_id:
            self.camera_canvas.delete(self.current_rect_id)
        self.current_rect_id = self.camera_canvas.create_rectangle(self.drag_start_x, self.drag_start_y, event.x, event.y, outline="red", width=2)

    def on_roi_dragging(self, event):
        if self.current_rect_id:
            self.camera_canvas.coords(self.current_rect_id, self.drag_start_x, self.drag_start_y, event.x, event.y)

    def on_roi_drag_end(self, event):
        end_x = event.x
        end_y = event.y
        
        # Reverse geometry indices safely if drag execution was handled backward
        x1, x2 = min(self.drag_start_x, end_x), max(self.drag_start_x, end_x)
        y1, y2 = min(self.drag_start_y, end_y), max(self.drag_start_y, end_y)
        
        canvas_w = self.camera_canvas.winfo_width()
        canvas_h = self.camera_canvas.winfo_height()
        
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return # Cancel calculation if click execution was a false positive jitter
            
        # Extrapolate canvas bounding coordinates back onto full resolution camera space (assuming 1920x1080 scaling limits)
        cam_max_w = 1920
        cam_max_h = 1080
        
        scaled_x = int((x1 / canvas_w) * cam_max_w)
        scaled_y = int((y1 / canvas_h) * cam_max_h)
        scaled_w = int(((x2 - x1) / canvas_w) * cam_max_w)
        scaled_h = int(((y2 - y1) / canvas_h) * cam_max_h)
        
        # Round boundaries to avoid VmbPy validation edge exceptions
        scaled_x = (scaled_x // 2) * 2
        scaled_y = (scaled_y // 2) * 2
        scaled_w = (scaled_w // 4) * 4
        scaled_h = (scaled_h // 4) * 4
        
        # Safely enforce minimum dimensions to prevent camera initialization crashes
        scaled_w = max(16, min(scaled_w, cam_max_w - scaled_x))
        scaled_h = max(16, min(scaled_h, cam_max_h - scaled_y))

        # Push calculated values out onto entry parameter bindings smoothly
        self.roi_x.set(scaled_x)
        self.roi_y.set(scaled_y)
        self.roi_w.set(scaled_w)
        self.roi_h.set(scaled_h)
        
        self.apply_manual_roi_parameters()

    # =========================================================================
    # MATRIX SUBTRACTOR PROFILES MANAGEMENT
    # =========================================================================

    def execute_immediate_snapshot(self):
        """Instantly snapshot and freeze a single frame without pulsing the ADS lines."""
        if self.latest_live_frame is not None:
            self.latest_pulsed_snapshot = self.latest_live_frame.copy()
            self.refresh_data_snapshot_viewports()

    def capture_background_profile(self):
        """Grabs the current raw frame and sets it as the active baseline subtraction background."""
        if self.latest_live_frame is not None:
            self.background_image = self.latest_live_frame.copy()
            self.refresh_data_snapshot_viewports()
            print("[Matrix System] Active reference background calibration profile locked down.")

    def clear_background_buffer(self):
        self.background_image = None
        self.refresh_data_snapshot_viewports()
        print("[Matrix System] Reference background array buffer cleared.")

    def refresh_data_snapshot_viewports(self):
        """Redraws the bottom right panels to display the original and processed background streams side by side."""
        w = self.lbl_snapshot_display.winfo_width()
        h = self.lbl_snapshot_display.winfo_height()
        if w < 10 or h < 10:
            w, h = 320, 240

        # Render Left Profile (Resulting Processed Snapshot)
        if self.latest_pulsed_snapshot is not None:
            processed = self.latest_pulsed_snapshot.copy()
            if self.background_image is not None and self.background_image.shape == processed.shape:
                processed = cv2.absdiff(processed, self.background_image)
                
            img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB) if len(processed.shape) == 3 else cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img_rgb).resize((w, h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            self._snap_tk_holder = tk_img
            self.lbl_snapshot_display.config(image=tk_img)
        else:
            self.lbl_snapshot_display.config(image="", text="No Snapshot Triggered Yet")

        # Render Right Profile (Active Background Matrix Reference)
        if self.background_image is not None:
            img_rgb = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2RGB) if len(self.background_image.shape) == 3 else cv2.cvtColor(self.background_image, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img_rgb).resize((w, h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            self._bg_tk_holder = tk_img
            self.lbl_background_display.config(image=tk_img)
        else:
            self.lbl_background_display.config(image="", text="Empty Background Frame Vector Buffer")

    def save_current_snapshot_manually(self):
        if self.latest_pulsed_snapshot is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image File", "*.png")])
            if path:
                cv2.imwrite(path, self.latest_pulsed_snapshot)
                print(f"[Storage Matrix] Manually written frame exported out to: {path}")

    # =========================================================================
    # THE MASTER SYNCHRONIZED TIMING PULSE SEQUENCE ENGINE
    # =========================================================================

    def execute_master_pulse_routine(self):
        """Coordinates multi-instrument synchronized execution across an asynchronous worker pool."""
        # Visual indicators locking downstream operations
        self.btn_master_pulse.config(text="RUNNING SEQUENCE...", state="disabled")
        self.live_view_active = False # Pause top-right continuous loop tracking

        def sequence_execution_worker():
            try:
                # 1. Gather all GUI operational parameters safely
                delay_to_snap_ms = self.var_time_after_pulse.get()
                pd3_domain_ms = self.var_pd3_window.get()
                total_pulses = self.var_num_pulses.get()
                step_len_ms = self.var_step_length.get()
                
                # Math translations scaling parameters to seconds
                delay_to_snap_s = delay_to_snap_ms / 1000.0
                pd3_domain_s = pd3_domain_ms / 1000.0
                
                print(f"[Pulse Engine] Commencing {total_pulses} synchronized hardware triggers...")

                # 2. Check and configure the Analog In (Oscilloscope) Instrument if hardware is connected
                if self.ads:
                    scope_sample_rate = 100000.0 # 100 kHz standard sampling requirement
                    scope_buffer_samples = int(scope_sample_rate * pd3_domain_s)
                    scope_buffer_samples = max(1024, scope_buffer_samples) # DWF requires clean base window scales
                    
                    self.ads.analog_in_reset()
                    self.ads.analog_in_channel_enable(channel=0, enable=True)
                    self.ads.analog_in_set_sample_rate(scope_sample_rate)
                    self.ads.analog_in_set_buffer_size(scope_buffer_samples)
                    
                    # Set up hardware triggering off the synchronized digital bus line
                    # trigsrcDigitalIn (3 or 5) gates the buffer storage sweep seamlessly
                    self.ads._dwf.FDwfAnalogInTriggerSourceSet(self.ads._hdwf, ctypes.c_byte(3)) 
                    self.ads._dwf.FDwfAnalogInTriggerTypeSet(self.ads._hdwf, ctypes.c_int(0)) # Edge triggering
                    self.ads._dwf.FDwfAnalogInTriggerConditionSet(self.ads._hdwf, ctypes.c_int(0)) # Rising edge trigger
                    
                    # Arm the oscilloscope engine to wait for the upcoming digital pulse
                    self.ads.analog_in_configure(reconfigure=True, start=True)

                # 3. Arm and prepare the Camera for the incoming hardware TTL trigger edge
                if self.camera and self.camera._cam:
                    # Configure camera to listen on Line 1 for a rising edge trigger
                    hw_trigger_config = HardwareTriggerConfig(
                        line="Line1",
                        activation=TriggerActivation.RISING_EDGE,
                        selector=TriggerSelector.FRAME_START,
                        acquisition_mode=AcquisitionMode.SINGLE_FRAME,
                        timeout_s=5.0
                    )
                    self.camera.arm_hardware_trigger(hw_trigger_config)

                # 4. Configure the hardware pattern generator lines via the custom controller
                if self.ttl_trig:
                    # Dynamically combine lines based on user settings checkboxes
                    active_pins = [1, 2] # Default to Shutter (DIO 1) and Camera Sync (DIO 2)
                    if self.var_sync_pulse.get():
                        active_pins.append(0) # Include Magnet (DIO 0) in the pattern block
                    
                    # Generate identical high/low phase times across the pattern bank
                    pulse_config = TTLTriggerConfig(
                        pins=active_pins,
                        high_time_s=pd3_domain_s,
                        low_time_s=self.var_time_between_pulses.get(),
                        pulse_count=total_pulses,
                        idle_state=TTLIdleState.LOW,
                        polarity=TTLPolarity.ACTIVE_HIGH,
                        mode=TTLTriggerMode.IMMEDIATE,
                        delay_s=0.0
                    )
                    
                    self.ttl_trig.configure(pulse_config)
                    
                    # Fire pattern generator lines directly
                    # Device timing executes via FPGA clock arrays down at the 10ns precision floor
                    self.ttl_trig.fire()

                # 5. Collect the data arrays captured by the external instruments concurrently
                # Catch the camera frame via the asynchronous worker thread
                captured_frame = None
                if self.camera and self.camera._cam:
                    try:
                        captured_frame = self.camera.wait_for_hardware_trigger(timeout_s=5.0)
                    except Exception as err:
                        print(f"[Pulse Engine Camera Exception] Match trace drop: {err}")
                    finally:
                        self.camera.disarm_hardware_trigger()

                # Poll and grab the oscilloscope data arrays
                scope_voltages = np.array([])
                if self.ads:
                    timeout_limit = time.time() + 5.0
                    while True:
                        status = self.ads.analog_in_status(read_data=True)
                        if status == 2: # DwfStateDone: Buffer collection completed
                            scope_voltages = self.ads.analog_in_read_data(channel=0, buffer_size=scope_buffer_samples)
                            break
                        if time.time() > timeout_limit:
                            print("[Pulse Engine Scope Timeout] Exceeded data collection window.")
                            break
                        time.sleep(0.01)

                # 6. Post-process the collected data arrays, update visualizations, and save files
                self.after(0, self.finalize_and_render_pulse_metrics, captured_frame, scope_voltages, pd3_domain_s)

            except Exception as outer_err:
                print(f"[Fatal Sequence Error] {outer_err}")
                self.after(0, self.reset_interface_execution_safeguards)

        threading.Thread(target=sequence_execution_worker, daemon=True).start()

    def finalize_and_render_pulse_metrics(self, captured_frame, scope_voltages, total_duration_s):
        """Brings the user interface out of freeze lock, saving array inputs out to stable disks."""
        if captured_frame is not None:
            self.latest_pulsed_snapshot = captured_frame.copy()
            self.refresh_data_snapshot_viewports()
            
            # Execute automated saving of snapshot files if authorized
            if self.var_auto_save_pulsed_img.get():
                filename = f"{self.var_pulse_filename_base.get()}_{int(time.time())}.png"
                cv2.imwrite(filename, captured_frame)
                print(f"[Auto-Save Matrix] Pulse snapshot saved: {filename}")

        if scope_voltages is not None and len(scope_voltages) > 0:
            self.render_oscilloscope_canvas_trace(scope_voltages)
            
            # Save the raw voltage trace array against its time indices to a CSV file if enabled
            if self.var_save_csv.get():
                csv_destination = self.var_csv_path.get()
                try:
                    time_steps = np.linspace(0, total_duration_s, len(scope_voltages))
                    with open(csv_destination, mode="w", newline="") as file_handle:
                        writer = csv.writer(file_handle)
                        writer.writerow(["Time Indices (s)", "Fluorescence (PD3 Voltage)"])
                        for t_idx, v_val in zip(time_steps, scope_voltages):
                            writer.writerow([t_idx, v_val])
                    print(f"[CSV Matrix Log] Trace log written out cleanly to: {csv_destination}")
                except Exception as csv_err:
                    print(f"[CSV File Export Error] Block write collapsed: {csv_err}")

        # Restore system state variables to resume normal live operations
        self.reset_interface_execution_safeguards()

    def render_oscilloscope_canvas_trace(self, voltage_array):
        """Draws the oscilloscope trace onto the canvas."""
        self.scope_canvas.delete("all")
        w = self.scope_canvas.winfo_width()
        h = self.scope_canvas.winfo_height()
        if w < 10 or h < 10:
            w, h = 500, 180

        # Draw grid reference lines
        self.scope_canvas.create_line(0, h//2, w, h//2, fill="#222222")
        
        if len(voltage_array) < 2:
            return

        # Normalize arrays values into geometric visual heights
        v_min, v_max = np.min(voltage_array), np.max(voltage_array)
        span = (v_max - v_min) if (v_max - v_min) > 0.01 else 1.0
        
        points = []
        for idx, val in enumerate(voltage_array):
            x_pixel = int((idx / len(voltage_array)) * w)
            # Center and fit the signal trace to the canvas height
            y_pixel = int(h - 20 - ((val - v_min) / span) * (h - 40))
            points.append((x_pixel, y_pixel))

        # Flatten point pairs and render a smooth line sequence
        flat_points = [coord for pt in points for coord in pt]
        self.scope_canvas.create_line(flat_points, fill="#00ff00", width=1.5)
        
        # Add tracking labels to the visual scale bounds
        self.scope_canvas.create_text(35, 15, text=f"Max: {v_max:.3f} V", fill="#888888", font=("Arial", 8))
        self.scope_canvas.create_text(35, h - 15, text=f"Min: {v_min:.3f} V", fill="#888888", font=("Arial", 8))

    def reset_interface_execution_safeguards(self):
        """Re-enables the GUI inputs and resumes the live video processing loop safely."""
        self.btn_master_pulse.config(text="FIRE MASTER PULSE", state="normal")
        self.live_view_active = True

if __name__ == "__main__":
    app = CoreInstrumentApplication()
    app.mainloop()