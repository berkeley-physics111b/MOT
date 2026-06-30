import os
import csv
import time
import ctypes
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

# Import the custom hardware wrappers provided in your environment
from waveforms_ads import (
    WaveFormsADS,
    DwfDigitalOutIdleLow, DwfDigitalOutIdleHigh,
    DwfStateDone,
    trigsrcDetectorDigitalIn,
)
from allied_vision_camera import AlliedVisionCamera, CameraConfig, HardwareTriggerConfig, TriggerActivation, TriggerSelector, AcquisitionMode

# The hardware wrapper modules (allied_vision_camera, ttl_trigger,
# waveforms_ads) attach a NullHandler to their own loggers so they stay
# silent when imported as libraries. Without a handler configured here,
# real warnings/errors raised inside those modules (bad ROI, dropped
# frames, GenICam feature failures, etc.) are logged and then simply
# vanish with no console output -- which is exactly what made the
# live-view bug below look like it was failing with "no errors".
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _normalize_camera_frame(frame):
    """
    VmbPy's Frame.as_numpy_ndarray() can return mono frames as a 3-D array
    shaped (H, W, 1) rather than a true 2-D (H, W) array, depending on SDK
    version / pixel format. Collapse that redundant trailing single-channel
    dimension so every frame flowing through this application has one
    predictable shape: 2-D for mono, 3-D (H, W, 3) for color. This keeps
    shape-based checks (channel detection, background-subtraction shape
    comparisons) correct regardless of which shape VmbPy happened to hand
    back for a given frame.
    """
    if frame is None:
        return None
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return frame


def _frame_to_display_rgb(frame):
    """
    Convert a camera frame into an RGB array suitable for PIL/Tk display.

    The previous logic picked BGR-vs-mono conversion using
    `len(frame.shape) == 3`, which broke as soon as a mono frame arrived
    shaped (H, W, 1): that's "3 dimensions" too, so it got routed into
    COLOR_BGR2RGB and OpenCV raised an "invalid number of channels" error
    on every single frame. This always normalizes first and dispatches on
    the actual channel count instead of guessing from ndim alone.
    """
    frame = _normalize_camera_frame(frame)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    channels = frame.shape[2]
    if channels == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported frame shape for display: {frame.shape}")


class CoreInstrumentApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MOT Control Panel")
        self.geometry("1400x900")
        self.state("zoomed") # Open maximized for visual real estate

        # Initialize Hardware Interface Objects
        self.ads = None
        self.camera = None
        
        # State Arrays and Image Matrices
        self.background_image = None
        self.latest_live_frame = None
        self.latest_pulsed_snapshot = None
        self.live_view_active = True

        # Persistent canvas image item id for the live view; reused via
        # itemconfig() on every frame instead of stacking a fresh
        # create_image() each time (which previously leaked one canvas
        # item per frame and degraded performance over time).
        self._live_canvas_image_id = None

        # Canvas Drag-to-ROI Coordinates
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.current_rect_id = None
        
        # Connect to Devices Safeguarded against immediate missing hardware
        self.init_hardware_connections()

        # Build GUI Frame Panes
        self.create_four_panels()
        
        # Start Live View Processing Loop
        self.start_live_view()

    def init_hardware_connections(self):
        """Secure safe handles to the hardware layers."""
        try:
            self.ads = WaveFormsADS()
            # FDwfDeviceEnableSet(1) is the master gate for ALL device
            # outputs (both Digital I/O static writes and the Digital Out
            # pattern generator). The waveforms_ads wrapper only flips this
            # via its outputs_enabled() context manager, which nothing in
            # this app uses -- so without this explicit call there is no
            # guarantee the master output stage is actually enabled.
            self.ads._dwf.FDwfDeviceEnableSet(self.ads._hdwf, 1)
            # Initialize DIO 0,1,2 as outputs explicitly if required by platform
            self.ads.digital_io_set_output_enable(0x07)
        except Exception as e:
            print(f"[Warning] Analog Discovery device could not connect: {e}")
            self.ads = None

        # Discovered hardware-trigger vocabulary for the connected camera.
        # Different Allied Vision models expose different GPIO line names
        # and trigger selector entries -- "Line1" / "FrameStart" are common
        # but NOT universal, and feeding an unsupported name to the camera
        # raises a GenICam "no enum entry" error. Query what this specific
        # camera actually supports instead of hardcoding a guess.
        self.available_trigger_lines = []
        self.available_trigger_selectors = []

        try:
            # Generate baseline configuration for Allied Vision
            cam_cfg = CameraConfig(exposure_time_us=20000, gain=0.0, brightness=0.0)
            self.camera = AlliedVisionCamera(cam_cfg)
            self.camera.open()

            try:
                self.available_trigger_lines = list(self.camera.list_hardware_trigger_lines())
                print(f"[Camera] GPIO trigger lines reported by this camera: {self.available_trigger_lines}")
            except Exception as e:
                print(f"[Camera] Could not enumerate GPIO trigger lines: {e}")

            try:
                self.available_trigger_selectors = [str(entry) for entry in self.camera._cam.TriggerSelector.get_all_entries()]
                print(f"[Camera] Trigger selectors reported by this camera: {self.available_trigger_selectors}")
            except Exception as e:
                print(f"[Camera] Could not enumerate trigger selectors: {e}")
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
        """Draws a multi-channel timing schematic with a real time axis and repeated pulses."""
        self.sequence_canvas.delete("all")
        w = self.sequence_canvas.winfo_width() if self.sequence_canvas.winfo_width() > 50 else 500
        h = 140

        try:
            snap_delay_ms   = self.var_time_after_pulse.get()
            pd3_domain_ms   = self.var_pd3_window.get()
            between_s       = self.var_time_between_pulses.get()
            num_pulses      = max(1, self.var_num_pulses.get())
        except Exception:
            return  # suppress entry-parsing hiccups during typing

        # Total time window to display (ms)
        between_ms   = between_s * 1000.0
        total_ms     = (pd3_domain_ms + between_ms) * num_pulses
        if total_ms <= 0:
            return

        # Layout constants
        LEFT_MARGIN  = 70   # px for channel labels
        RIGHT_MARGIN = 10
        TOP_MARGIN   = 8
        AXIS_HEIGHT  = 18   # px for the time axis at the bottom
        plot_w = w - LEFT_MARGIN - RIGHT_MARGIN
        plot_h = h - TOP_MARGIN - AXIS_HEIGHT

        # Three channels; each gets 1/3 of plot_h
        ch_h     = plot_h // 3
        channels = [
            ("Mag DIO0", "#4caf50",  0),
            ("Sht DIO1", "#2196f3",  1),
            ("Cam DIO2", "#ff9800",  2),
        ]

        def t2x(t_ms):
            return LEFT_MARGIN + (t_ms / total_ms) * plot_w

        def row_y(row):
            """Return the baseline y for channel row (0-based)."""
            return TOP_MARGIN + row * ch_h + ch_h

        # Draw channel labels and baseline
        for row, (label, color, _pin) in enumerate(channels):
            base_y = row_y(row)
            self.sequence_canvas.create_text(
                LEFT_MARGIN - 4, base_y - ch_h // 2,
                text=label, fill=color, font=("Consolas", 8), anchor="e"
            )
            self.sequence_canvas.create_line(
                LEFT_MARGIN, base_y, w - RIGHT_MARGIN, base_y,
                fill="#2a2a2a", dash=(3, 4)
            )

        # Draw pulses for each channel
        for row, (label, color, pin) in enumerate(channels):
            base_y = row_y(row)
            high_y = base_y - int(ch_h * 0.75)

            t = 0.0
            # Lead-in flat line
            x0 = t2x(0)
            for p in range(num_pulses):
                # Rising edge
                x_rise = t2x(t)
                # High phase
                t_fall = t + pd3_domain_ms
                x_fall = t2x(t_fall)
                # Falling edge, then low until next pulse
                t_next = t + pd3_domain_ms + between_ms
                x_next = t2x(min(t_next, total_ms))

                # Camera sync pin (DIO 2) fires at snap_delay_ms after pulse start
                if pin == 2:
                    t_cam_rise = t + snap_delay_ms
                    t_cam_fall = t_cam_rise + min(5.0, pd3_domain_ms * 0.1)  # narrow blip
                    if t_cam_rise < total_ms:
                        xc0 = t2x(t_cam_rise)
                        xc1 = t2x(min(t_cam_fall, total_ms))
                        # flat low before blip
                        self.sequence_canvas.create_line(x0, base_y, xc0, base_y, fill=color, width=1)
                        # rising
                        self.sequence_canvas.create_line(xc0, base_y, xc0, high_y, fill=color, width=1)
                        # high
                        self.sequence_canvas.create_line(xc0, high_y, xc1, high_y, fill=color, width=1)
                        # falling
                        self.sequence_canvas.create_line(xc1, high_y, xc1, base_y, fill=color, width=1)
                        x0 = xc1
                else:
                    # flat low before rise
                    self.sequence_canvas.create_line(x0, base_y, x_rise, base_y, fill=color, width=1)
                    # rising edge
                    self.sequence_canvas.create_line(x_rise, base_y, x_rise, high_y, fill=color, width=1)
                    # high phase
                    self.sequence_canvas.create_line(x_rise, high_y, x_fall, high_y, fill=color, width=1)
                    # falling edge
                    self.sequence_canvas.create_line(x_fall, high_y, x_fall, base_y, fill=color, width=1)
                    x0 = x_fall

                t = t_next

            # Tail flat line to end
            x_end = t2x(total_ms)
            self.sequence_canvas.create_line(x0, base_y, x_end, base_y, fill=color, width=1)

        # Time axis
        axis_y = TOP_MARGIN + plot_h + 2
        self.sequence_canvas.create_line(
            LEFT_MARGIN, axis_y, w - RIGHT_MARGIN, axis_y, fill="#555555", width=1
        )

        # Tick marks: aim for ~5 ticks
        n_ticks = 5
        tick_step_ms = total_ms / n_ticks
        # Round to a nice number
        for scale in [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
            if scale >= tick_step_ms * 0.5:
                tick_step_ms = scale
                break

        t_tick = 0.0
        while t_tick <= total_ms + tick_step_ms * 0.01:
            xt = t2x(min(t_tick, total_ms))
            self.sequence_canvas.create_line(xt, axis_y, xt, axis_y + 4, fill="#555555")
            if t_tick < 1000:
                lbl = f"{t_tick:.0f}ms" if t_tick == int(t_tick) else f"{t_tick:.1f}ms"
            else:
                lbl = f"{t_tick/1000:.1f}s"
            self.sequence_canvas.create_text(
                xt, axis_y + 10, text=lbl, fill="#666666", font=("Consolas", 7), anchor="n"
            )
            t_tick += tick_step_ms

    def build_top_right_panel(self):
        """Live video viewport and camera configuration settings controls."""
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

        # Hardware Trigger Wiring -- populated from whatever GPIO lines and
        # trigger selectors THIS connected camera actually reports
        # (queried in init_hardware_connections). Different camera models
        # use different names here ("Line1" / "FrameStart" are common
        # defaults but not universal), so hardcoding them caused
        # "no enum entry" errors on cameras that don't expose those exact
        # names. Falling back to a placeholder list keeps the GUI usable
        # even if no camera is connected yet.
        line_options = self.available_trigger_lines or ["Line1"]
        selector_options = self.available_trigger_selectors or ["FrameStart"]

        self.var_trigger_line = tk.StringVar(value=line_options[0])
        self.var_trigger_selector = tk.StringVar(value=selector_options[0])

        trig_box = ttk.LabelFrame(controls_frame, text="Hardware Trigger Wiring")
        trig_box.pack(fill="x", pady=4, padx=2)

        ttk.Label(trig_box, text="GPIO Line:").grid(row=0, column=0, **grid_params)
        ttk.Combobox(trig_box, textvariable=self.var_trigger_line, values=line_options, width=14, state="readonly").grid(row=0, column=1, sticky="w")

        ttk.Label(trig_box, text="Trigger Selector:").grid(row=1, column=0, **grid_params)
        ttk.Combobox(trig_box, textvariable=self.var_trigger_selector, values=selector_options, width=14, state="readonly").grid(row=1, column=1, sticky="w")

        if not self.available_trigger_lines or not self.available_trigger_selectors:
            ttk.Label(trig_box, text="(No camera connected -- placeholder values shown)", foreground="#cc8800").grid(row=2, column=0, columnspan=2, sticky="w", padx=2)

        # Operational Control Buttons Pack
        action_box = ttk.LabelFrame(controls_frame, text="Direct Trigger Operations")
        action_box.pack(fill="x", pady=4, padx=2)

        ttk.Button(action_box, text="Capture Snapshot Now", command=self.execute_immediate_snapshot).pack(fill="x", pady=2)
        ttk.Button(action_box, text="Extract & Save Background", command=self.capture_background_profile).pack(fill="x", pady=2)
        
        self.btn_synch_pulse = tk.Button(action_box, text="Pulse", bg="#2f2525", fg="white", font=("Arial", 11, "bold"), command=self.execute_synch_pulse_routine)
        self.btn_synch_pulse.pack(fill="x", pady=6)

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
            # ROI changes are commonly rejected or silently ignored by
            # GenICam cameras while continuous streaming is active (the
            # underlying SDK even warns about this in set_roi()). Pause the
            # live view, apply the change, then resume it.
            self._stop_camera_live_view()

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
        finally:
            self._start_camera_live_view()

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
        """Drives manual state level adjustments across the static digital registers.

        DIO 0 is the magnet coil.  Setting it high holds the magnet on
        indefinitely; clearing it turns it off.  We use digital_io_write_pin()
        (read-modify-write on the current output register) so we never disturb
        the state of any other pin.

        The Digital I/O write itself is instant, but we offload it to a
        background thread anyway so that any unexpected ADS latency cannot
        freeze the Tkinter main loop.
        """
        if not self.ads:
            return

        state_bit = bool(self.var_magnet_state.get())

        def _write():
            try:
                # Ensure DIO 0-2 are still configured as outputs (a previous
                # digital_out_reset() or digital_io_reset() call inside the
                # pulse sequence can clear the output-enable mask).
                self.ads.digital_io_set_output_enable(0x07)
                self.ads.digital_io_write_pin(pin=0, value=state_bit)
                print(f"[Magnet] DIO 0 set {'HIGH (on)' if state_bit else 'LOW (off)'}")
            except Exception as e:
                print(f"[Error] Failed magnet register write: {e}")

        threading.Thread(target=_write, daemon=True).start()

    def browse_csv_destination_file(self):
        """Launches localized system directory browser to identify data dump locations."""
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Tables", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.var_csv_path.set(file_path)

    # =========================================================================
    # LIVE VIEW STREAM PROCESSING LOOP
    # =========================================================================

    def start_live_view(self):
        """
        Start continuous (free-run) acquisition for the live display.

        The previous implementation polled `camera.take_snapshot()` at
        ~30 Hz from a manual thread. take_snapshot() is the low-latency
        *software-triggered single-shot* path: every call reconfigures
        TriggerSource/TriggerSelector/TriggerMode/AcquisitionMode and runs a
        full start_streaming()/stop_streaming() cycle -- far too much
        per-frame GenICam overhead for live display, and prone to
        intermittent failures under that load. AlliedVisionCamera already
        exposes start_continuous()/stop_continuous() for exactly this case
        (free-run, non-critical timing), so we use that and let VmbPy's own
        streaming thread hand us frames via callback instead.
        """
        self._start_camera_live_view()

    def _start_camera_live_view(self):
        """Arm continuous (free-run) streaming and register the frame callback."""
        if not (self.camera and self.camera._cam):
            return
        try:
            self.camera.start_continuous(callback=self._on_live_frame)
        except Exception as e:
            print(f"[Live View] Could not start continuous streaming: {e}")

    def _stop_camera_live_view(self):
        """Stop continuous streaming. Required before any software/hardware trigger use."""
        if not (self.camera and self.camera._cam):
            return
        try:
            self.camera.stop_continuous()
        except Exception as e:
            print(f"[Live View] Could not stop continuous streaming: {e}")

    def _on_live_frame(self, raw_frame, timestamp_s):
        """
        Frame callback invoked by VmbPy's internal streaming thread -- this
        is NOT the Tkinter main thread. Tk/Tcl is not thread-safe, so no Tk
        widget calls (winfo_*, Canvas/PhotoImage creation, etc.) may happen
        here. Do the lightweight numpy bookkeeping on this thread, then hand
        off to the main thread via after() for anything GUI-related.
        """
        if not self.live_view_active:
            return
        try:
            frame = _normalize_camera_frame(raw_frame)
            self.latest_live_frame = frame.copy()
            self.after(0, self._update_live_canvas, frame)
        except Exception as e:
            # The old polling loop wrapped this whole path in a bare
            # `except Exception: pass`, so failures like this (e.g. the
            # channel-shape bug below) fired on every frame and never
            # produced a single line of console output.
            print(f"[Live View] Frame processing error: {e}")

    def _update_live_canvas(self, raw_frame):
        """
        Main-thread-only: apply background subtraction, convert to RGB,
        scale to the canvas size, and paint it. All Tk widget calls live
        here, since this only ever runs via self.after() on the main loop.
        """
        if not self.live_view_active:
            return

        processed_frame = raw_frame
        if self.background_image is not None and self.background_image.shape == processed_frame.shape:
            processed_frame = cv2.absdiff(processed_frame, self.background_image)

        canvas_w = self.camera_canvas.winfo_width()
        canvas_h = self.camera_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 640, 480

        try:
            img_rgb = _frame_to_display_rgb(processed_frame)
        except Exception as e:
            print(f"[Live View] Could not convert frame for display: {e}")
            return

        pil_img = Image.fromarray(img_rgb).resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self._render_tk_image_to_canvas(tk_img)

    def _render_tk_image_to_canvas(self, tk_img):
        self._live_tk_image_holder = tk_img  # Maintain pointer memory reference to prevent sudden garbage collection drops

        # Reuse a single canvas image item via itemconfig() instead of
        # calling create_image() on every frame. The original version
        # created a brand-new image item ~30 times/second without ever
        # deleting the previous one, silently accumulating thousands of
        # canvas items per session until the UI bogged down.
        if self._live_canvas_image_id is None:
            self._live_canvas_image_id = self.camera_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        else:
            self.camera_canvas.itemconfig(self._live_canvas_image_id, image=tk_img)

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

            try:
                img_rgb = _frame_to_display_rgb(processed)
                pil_img = Image.fromarray(img_rgb).resize((w, h), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(image=pil_img)
                self._snap_tk_holder = tk_img
                self.lbl_snapshot_display.config(image=tk_img)
            except Exception as e:
                print(f"[Snapshot Viewport] Could not display pulsed snapshot: {e}")
        else:
            self.lbl_snapshot_display.config(image="", text="No Snapshot Triggered Yet")

        # Render Right Profile (Active Background Matrix Reference)
        if self.background_image is not None:
            try:
                img_rgb = _frame_to_display_rgb(self.background_image)
                pil_img = Image.fromarray(img_rgb).resize((w, h), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(image=pil_img)
                self._bg_tk_holder = tk_img
                self.lbl_background_display.config(image=tk_img)
            except Exception as e:
                print(f"[Snapshot Viewport] Could not display background frame: {e}")
        else:
            self.lbl_background_display.config(image="", text="Empty Background Frame Vector Buffer")

    def save_current_snapshot_manually(self):
        if self.latest_pulsed_snapshot is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image File", "*.png")])
            if path:
                cv2.imwrite(path, self.latest_pulsed_snapshot)
                print(f"[Storage Matrix] Manually written frame exported out to: {path}")
        else:
            print(f"[Storage Matrix] No pulsed snapshot to save")

    # =========================================================================
    # THE SYNCHRONIZED TIMING PULSE SEQUENCE ENGINE
    # =========================================================================

    def execute_synch_pulse_routine(self):
        """Coordinates multi-instrument synchronized execution across an asynchronous worker pool."""
        # Visual indicators locking downstream operations
        self.btn_synch_pulse.config(text="RUNNING SEQUENCE...", state="disabled")
        self.live_view_active = False # Pause top-right continuous loop tracking

        # take_snapshot()/arm_hardware_trigger() both raise if continuous
        # streaming is still active, so stop it here on the main thread
        # before the worker thread below starts touching the camera.
        self._stop_camera_live_view()

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

                # Scope buffer size -- computed early so the variable is always
                # in scope for the data-read block at the bottom.
                scope_sample_rate    = 100000.0  # 100 kHz
                scope_buffer_samples = max(1024, int(scope_sample_rate * pd3_domain_s))

                # 2. Program the Digital Out pattern generator.
                #
                # digital_out_pulse_train() calls digital_out_reset() internally,
                # which on some ADS firmware revisions disturbs other instruments.
                # We do this BEFORE arming the oscilloscope so the reset can't
                # clobber the scope arm that follows.  We pass wait_for_done=False
                # here and manage completion ourselves via a threading.Event below.
                #
                # Pin roles:
                #   DIO 0 – magnet coil  (only when "Synchronize" checkbox is on)
                #   DIO 1 – shutter
                #   DIO 2 – camera sync
                #
                # high_time_s  = duration the pin stays HIGH per pulse (pd3 window)
                # low_time_s   = idle time BETWEEN pulses; the pulse train is
                #                HIGH for pd3_domain_s then LOW for time_between_pulses,
                #                repeated total_pulses times.
                pulse_done_event = threading.Event()
                pulse_error      = [None]

                if self.ads:
                    between_s   = self.var_time_between_pulses.get()
                    pulse_pins  = [1, 2]
                    pulse_highs = [pd3_domain_s, pd3_domain_s]
                    pulse_lows  = [between_s, between_s]

                    if self.var_sync_pulse.get():
                        pulse_pins.insert(0, 0)
                        pulse_highs.insert(0, pd3_domain_s)
                        pulse_lows.insert(0, between_s)

                    total_run_s   = (pd3_domain_s + between_s) * total_pulses
                    pulse_timeout = total_run_s + 1.0

                    print(f"[Pulse Engine] Programming digital_out: pins={pulse_pins} "
                          f"high={pd3_domain_s*1000:.1f}ms low={between_s*1000:.0f}ms "
                          f"x{total_pulses} (run={total_run_s:.3f}s)")

                    # Program each channel manually (mirrors what
                    # digital_out_pulse_train() does internally) so we can
                    # control the exact call order and force the trigger
                    # source to "none" (immediate) right before configure().
                    #
                    # WHY NOT JUST CALL digital_out_pulse_train()?
                    # That helper calls digital_out_reset() first, then
                    # programs each channel, then calls
                    # digital_out_configure(start=True) at the very end --
                    # but it never explicitly sets the trigger source. If a
                    # previous session (or another instrument on this device)
                    # left TriggerSource pointing at something other than
                    # trigsrcNone, configure(start=True) puts the generator
                    # into Armed/Wait state waiting for a trigger edge that
                    # never arrives. No exception is raised -- the pins
                    # simply never move, and digital_out_status() sits at
                    # Armed/Wait until our host-side timeout silently expires.
                    # That exactly matches the symptom reported ("no errors,
                    # pins not moving").
                    self.ads.digital_out_reset()

                    clk = self.ads.digital_out_get_internal_clock()
                    MIN_TICKS, MAX_COUNT = 1, 0xFFFF_FFFF

                    for pin, ht, lt in zip(pulse_pins, pulse_highs, pulse_lows):
                        total_high = max(MIN_TICKS, round(clk * ht))
                        total_low  = max(MIN_TICKS, round(clk * lt))
                        divider = 1
                        while (total_high // divider > MAX_COUNT or
                               total_low  // divider > MAX_COUNT):
                            divider += 1
                        high_ticks = max(1, round(total_high / divider))
                        low_ticks  = max(1, round(total_low  / divider))

                        self.ads.digital_out_enable_channel(pin, True)
                        self.ads.digital_out_set_output_mode(pin, 0)  # push-pull
                        self.ads.digital_out_set_type(pin, 0)         # pulse
                        self.ads.digital_out_set_idle(pin, DwfDigitalOutIdleLow)
                        self.ads.digital_out_set_divider_init(pin, divider)
                        self.ads.digital_out_set_divider(pin, divider)
                        self.ads.digital_out_set_counter_init(pin, start_high=True, initial_count=high_ticks)
                        self.ads.digital_out_set_counter(pin, low_count=low_ticks, high_count=high_ticks)

                    # Global timing
                    self.ads.digital_out_set_wait_time(0.0)
                    self.ads.digital_out_set_run_time(total_run_s)
                    self.ads.digital_out_set_repeat(1)
                    try:
                        self.ads.digital_out_set_repeat_trigger(False)
                    except Exception:
                        pass  # not all firmware revisions expose this

                    # Force immediate (untriggered) start -- see note above.
                    self.ads.digital_out_set_trigger_source(0)  # trigsrcNone

                    self.ads.digital_out_configure(start=True)
                    print(f"[Pulse Engine] digital_out_configure(start=True) issued; "
                          f"status={self.ads.digital_out_status()}")

                    # Watcher thread: poll for Done state without blocking the
                    # rest of the sequence (scope/camera run concurrently).
                    def _wait_pulse_done():
                        try:
                            deadline = time.time() + pulse_timeout
                            last_status = None
                            while True:
                                status = self.ads.digital_out_status()
                                if status != last_status:
                                    print(f"[Pulse Engine] digital_out status -> {status}")
                                    last_status = status
                                if status == DwfStateDone:
                                    print("[Pulse Engine] digital_out Done.")
                                    break
                                if time.time() > deadline:
                                    pulse_error[0] = TimeoutError(
                                        f"Pulse did not finish within {pulse_timeout:.1f}s "
                                        f"(stuck in status={status} -- if this is 1 (Armed) or "
                                        f"7 (Wait), the generator is waiting on a trigger that "
                                        f"never arrived)"
                                    )
                                    break
                                time.sleep(0.005)
                        except Exception as _e:
                            pulse_error[0] = _e
                        finally:
                            pulse_done_event.set()

                    threading.Thread(target=_wait_pulse_done, daemon=True).start()
                else:
                    # No ADS connected; signal immediately so the rest of the
                    # sequence doesn't block forever.
                    pulse_done_event.set()

                # 3. NOW arm the oscilloscope (after digital_out is already
                # programmed and running, so no subsequent reset can hit it).
                if self.ads:
                    self.ads.analog_in_reset()
                    self.ads.analog_in_channel_enable(channel=0, enable=True)
                    self.ads.analog_in_set_sample_rate(scope_sample_rate)
                    self.ads.analog_in_set_buffer_size(scope_buffer_samples)
                    # Trigger on the rising edge of the Digital Out bus
                    self.ads.analog_in_set_trigger_source(trigsrcDetectorDigitalIn)
                    self.ads.analog_in_set_trigger_type(0)       # edge
                    self.ads.analog_in_set_trigger_condition(0)  # rising
                    self.ads.analog_in_configure(reconfigure=True, start=True)
                    print(f"[Pulse Engine] Oscilloscope armed: {scope_buffer_samples} "
                          f"samples @ {scope_sample_rate/1e3:.0f} kHz")

                # 4. Take camera snapshot while pulse is running.
                captured_frame = None
                if self.camera and self.camera._cam:
                    try:
                        # TODO: switch to hardware trigger when wiring is confirmed
                        captured_frame = self.camera.take_snapshot()
                    except Exception as err:
                        print(f"[Pulse Engine Camera] Snapshot failed: {err}")

                # 5. Wait for the pulse train to complete, then read scope data.
                _wait_timeout = (total_run_s + 2.0) if self.ads else 1.0
                pulse_done_event.wait(timeout=_wait_timeout)
                if pulse_error[0]:
                    print(f"[Pulse Engine] Pulse error: {pulse_error[0]}")

                # After Digital Out finishes, restore DIO 0 (magnet) to
                # whatever the checkbox says -- the pulse temporarily overrides
                # the static Digital I/O level on that pin.
                if self.ads and self.var_sync_pulse.get():
                    try:
                        self.ads.digital_io_set_output_enable(0x07)
                        self.ads.digital_io_write_pin(
                            pin=0, value=bool(self.var_magnet_state.get())
                        )
                    except Exception as _e:
                        print(f"[Pulse Engine] Could not restore magnet state: {_e}")

                # Poll and grab the oscilloscope data arrays
                scope_voltages = np.array([])
                if self.ads:
                    timeout_limit = time.time() + 5.0
                    while True:
                        status = self.ads.analog_in_status(read_data=True)
                        if status == 2:  # DwfStateDone
                            scope_voltages = self.ads.analog_in_get_data(
                                channel=0, n_samples=scope_buffer_samples
                            )
                            print(f"[Pulse Engine] Scope captured {len(scope_voltages)} samples, "
                                  f"range [{scope_voltages.min():.3f}, {scope_voltages.max():.3f}] V")
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
            captured_frame = _normalize_camera_frame(captured_frame)
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
        self.btn_synch_pulse.config(text="PULSE", state="normal")
        self.live_view_active = True
        self._start_camera_live_view()

if __name__ == "__main__":
    app = CoreInstrumentApplication()
    app.mainloop()