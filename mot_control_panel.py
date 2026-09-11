import os
import csv
import time
import logging
import threading
from collections import deque
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
    trigsrcDigitalOut,
    funcDC, AnalogOutNodeCarrier,
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

        # True while a pulse sequence's background worker thread is
        # actively talking to self.ads / self.camera (arming triggers,
        # polling the scope, etc.). Neither hardware wrapper is
        # thread-safe against its own close() being called concurrently,
        # so on_app_close() must wait for this to clear before tearing
        # down any device handles -- see on_app_close() for details.
        self._sequence_running = False
        self._shutdown_waiting_on_sequence = False
        # Hard cap on how long on_app_close() will wait on a running pulse
        # sequence before giving up and tearing down hardware anyway. Set
        # the first time we notice a sequence is in flight during shutdown.
        self._shutdown_wait_deadline = None

        # Persistent canvas image item id for the live view; reused via
        # itemconfig() on every frame instead of stacking a fresh
        # create_image() each time (which previously leaked one canvas
        # item per frame and degraded performance over time).
        self._live_canvas_image_id = None

        # Canvas Drag-to-ROI Coordinates
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.current_rect_id = None

        # --- Laser Lock tab state (ADS Scope Ch1/Ch2 -> WaveGen Ch1) ---
        # Channel indices in the ADS API are 0-based; the physically
        # labeled "Ch1"/"Ch2" scope inputs and "Channel 1" AWG output
        # correspond to indices 0, 1, and 0 respectively.
        #   Scope Ch1 (index 0) = PD2, plain fluorescence/absorption signal.
        #   Scope Ch2 (index 1) = PD1b - PD1a, DAVLL error signal.
        #   WaveGen Ch1 (index 0) = laser modulation output (sweep + offset
        #   + internal bias + feedback, summed -- see _laser_lock_step).
        self.SCOPE_CH1_FLUOR = 0
        self.SCOPE_CH2_ERROR = 1
        self.WAVEGEN_MOD_CHANNEL = 0

        # Live scope readings, written only by the background loop thread
        # and read only by the polling GUI refresh callback below -- a
        # plain float/None assignment is atomic under the GIL, so no extra
        # locking is needed for this single-writer/single-reader pattern.
        # Ch1 is forced back to None whenever the lock is engaged, since
        # this tab stops sampling that channel while locked (freeing it up
        # for the other MOT tab that also needs it -- see _laser_lock_step).
        self._last_scope_ch1_v = None
        self._last_scope_ch2_v = None

        # Rolling (t, v) trace buffers behind the Ch1/Ch2 plots. Appended
        # to only by the background loop thread; the GUI refresh callback
        # only ever reads a list() snapshot of these for drawing, so it
        # can't be corrupted by a concurrent append (deque.append is
        # atomic under the GIL).
        self.SCOPE_TRACE_MAXLEN = 500
        self._scope_ch1_trace = deque(maxlen=self.SCOPE_TRACE_MAXLEN)
        self._scope_ch2_trace = deque(maxlen=self.SCOPE_TRACE_MAXLEN)

        # PID state -- operates on the bias-shifted Ch2 error signal (see
        # _laser_lock_step).
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = None

        # Reference time base for the software-generated sweep waveform.
        self._sweep_t0 = time.perf_counter()

        self._laser_lock_loop_period_s = 0.002  # best-effort ~500 Hz update rate
        # Signaled on app shutdown so the loop thread stops touching
        # self.ads before the device handle gets torn down.
        self._lock_stop_event = threading.Event()
        self._lock_thread = None
        # Cleared on app shutdown so the GUI-refresh self.after() chain
        # stops rescheduling itself against widgets that are about to be
        # destroyed.
        self._laser_lock_gui_alive = True

        # Connect to Devices Safeguarded against immediate missing hardware
        self.init_hardware_connections()

        # Build GUI Frame Panes
        self.create_four_panels()

        # Start the Laser Lock feedback-loop background thread. This runs
        # for the entire lifetime of the app -- not just while its tab
        # happens to be the one showing -- so the sweep/lock keeps running
        # no matter which tab the user has switched to. It only reads the
        # tk.Variables built in build_laser_lock_panel() and never touches
        # any widget directly, which is what makes it safe to run
        # regardless of tab visibility.
        self._lock_thread = threading.Thread(target=self._laser_lock_loop_worker, daemon=True)
        self._lock_thread.start()

        # Low-rate GUI refresh for the Scope Ch1/Ch2 live readouts. This
        # runs on the main thread via self.after and only ever touches Tk
        # widgets/variables -- it never talks to self.ads directly, so it
        # can't race with the background loop thread above.
        self.after(150, self._laser_lock_refresh_display)

        # Start Live View Processing Loop
        self.start_live_view()

        # Intercept the window manager's close button ("X") -- by default
        # Tk's WM_DELETE_WINDOW just destroys the window without giving the
        # application a chance to run any cleanup, so the camera stream and
        # ADS (WaveForms) device handles were previously left open,
        # sometimes forcing a hardware unplug/replug before the next run
        # would connect cleanly. Route the close button through an explicit
        # shutdown routine instead.
        self.protocol("WM_DELETE_WINDOW", self.on_app_close)

    def _run_with_timeout(self, fn, timeout_s, label):
        """
        Run fn() on a daemon thread and wait up to timeout_s for it to
        finish. If it doesn't finish in time, give up waiting and return
        anyway so shutdown can proceed -- the underlying thread is left
        running (as a daemon it can't block process exit), but critically
        this means a single stuck vendor SDK call (VmbPy stop_streaming(),
        Camera.__exit__(), VmbSystem.__exit__(), etc.) can no longer block
        every teardown step that comes after it, most importantly the
        Analog Discovery close() below. Without this, a hang anywhere in
        the camera teardown chain meant self.ads.close() was never reached
        at all, which is what previously required an ADS power cycle.
        """
        done = threading.Event()

        def _wrapped():
            try:
                fn()
            except Exception as e:
                print(f"[Shutdown] {label} raised: {e}")
            finally:
                done.set()

        threading.Thread(target=_wrapped, daemon=True).start()
        if not done.wait(timeout=timeout_s):
            print(f"[Shutdown] {label} did not complete within {timeout_s}s -- "
                  f"continuing shutdown anyway (likely stuck in a vendor SDK call).")

    def on_app_close(self):
        """
        Cleanly release all hardware handles before the window closes.

        Order matters here: streaming/acquisition must be stopped before a
        device is closed, and the camera's live-view thread must be told to
        stop pushing frames onto the Tk main loop (via self.after) before
        the window is torn down, or a queued callback can fire against
        widgets that no longer exist.
        """
        # A pulse sequence's worker thread (see execute_synch_pulse_routine)
        # actively drives self.ads / self.camera from a background thread
        # with no locking shared against close(). Closing those devices out
        # from under an in-flight hardware call is what previously froze
        # the app on shutdown -- the vendor SDK calls block forever waiting
        # on a device handle that's being torn down mid-operation. Instead
        # of closing immediately, re-check on the Tk main loop every 150 ms
        # until the worker thread signals it's done (reset_interface_
        # execution_safeguards clears self._sequence_running). This keeps
        # the GUI responsive rather than blocking, and is bounded because
        # every wait inside the worker thread has its own timeout (5-10 s).
        #
        # FIX 2: that per-wait timeout inside the worker is only as good as
        # the vendor SDK actually honoring it -- if a DWF/VmbPy call hangs
        # past its own stated timeout, _sequence_running could in principle
        # never clear and this branch would poll forever. Cap the total
        # time on_app_close() will wait here so shutdown is always bounded,
        # even in that worst case; if the deadline passes we give up
        # waiting and fall through to hardware teardown regardless of
        # whether the worker thread is technically still running.
        if self._sequence_running:
            if not self._shutdown_waiting_on_sequence:
                self._shutdown_waiting_on_sequence = True
                self._shutdown_wait_deadline = time.time() + 15.0
                print("[Shutdown] Pulse sequence still running -- waiting for it to finish before releasing hardware...")
                self.title("MOT Control Panel (finishing pulse sequence before closing...)")
            elif time.time() > self._shutdown_wait_deadline:
                print("[Shutdown] Pulse sequence wait exceeded 15s -- giving up waiting "
                      "and forcing hardware teardown anyway.")
                self._sequence_running = False

            if self._sequence_running:
                self.after(150, self.on_app_close)
                return

        print("[Shutdown] Close requested -- releasing hardware handles...")

        # Stop the live-view callback path first so no further frames get
        # scheduled onto this (soon to be destroyed) Tk main loop.
        self.live_view_active = False

        # --- Laser Lock feedback-loop thread ---
        # Signal the background loop to stop and give it a brief window to
        # notice, so it isn't still calling into self.ads (analog_in_read_
        # sample / analog_out_*) while the Analog Discovery handle below is
        # being closed out from under it. Also stop the GUI-refresh
        # self.after() chain so it doesn't reschedule itself against
        # widgets that are about to be destroyed.
        self._laser_lock_gui_alive = False
        self._lock_stop_event.set()
        if self._lock_thread is not None:
            self._lock_thread.join(timeout=1.5)

        # --- Camera ---
        # FIX 1: every one of these VmbPy calls (stop_streaming(),
        # Camera.__exit__(), VmbSystem.__exit__() inside camera.close())
        # is a blocking call into the vendor driver with no host-side
        # timeout of its own. Previously, if any of them hung (e.g. after
        # a hardware trigger that never fired), execution never reached
        # the Analog Discovery teardown below at all, since it runs
        # strictly after this block finishes -- that's what forced an ADS
        # power cycle even though the ADS itself wasn't the problem.
        # Running each step through _run_with_timeout() bounds the total
        # time this block can take, so the ADS teardown always runs.
        if self.camera is not None:
            self._run_with_timeout(self._stop_camera_live_view, 3.0, "camera stop_continuous")
            self._run_with_timeout(self.camera.close, 5.0, "camera.close")
            self.camera = None

        # --- Analog Discovery / WaveForms device ---
        if self.ads is not None:
            def _ads_teardown():
                # Leave the digital outputs in a known-safe (all-low) state
                # before tearing down the device handle, rather than
                # abandoning them mid-pulse if a sequence happened to be
                # interrupted.
                try:
                    # FIX 2: the Digital Out pattern generator (used for the
                    # DIO1/DIO2 pulse train, and DIO0 too when "Synchronize
                    # Magnet" is on) was previously never reset on shutdown
                    # -- only analog_in_reset() was called. If a sequence
                    # exited abnormally (worker exception, hardware trigger
                    # timeout, etc.) with the pattern generator still armed,
                    # closing the device handle on top of that left the ADS
                    # in a state that needed a power cycle to clear. Reset
                    # it explicitly here before anything else.
                    self.ads.digital_out_reset()
                except Exception:
                    pass
                try:
                    self.ads.digital_io_set_output_enable(0x01)
                    self.ads.digital_io_write_pin(pin=0, value=False)
                except Exception:
                    pass
                try:
                    self.ads.analog_in_reset()
                except Exception:
                    pass
                try:
                    # Leave the PID modulation output in a known-safe
                    # (disabled, 0 V) state rather than abandoning it at
                    # whatever correction voltage it last held.
                    self.ads.analog_out_reset(self.WAVEGEN_MOD_CHANNEL)
                except Exception:
                    pass
                self.ads.close()

            self._run_with_timeout(_ads_teardown, 3.0, "ads teardown")
            self.ads = None

        # Tear down the Tk event loop and window.
        try:
            self.quit()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass
        print("[Shutdown] Application closed.")

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
            # IMPORTANT: Digital I/O (static) and Digital Out (pattern
            # generator) are separate ADS sub-systems that share the same
            # physical DIO header. Whichever instrument claims a pin's
            # output-enable wins control of it; the Digital Out engine can
            # successfully program and run a pulse on a pin while reporting
            # Done, and still produce zero volts on the header if the
            # Digital I/O instrument also has that pin's output-enable bit
            # set (it will just keep driving its own static level).
            #
            # DIO 0 is the magnet coil and is controlled via static
            # digital_io_write_pin() -- it stays claimed by Digital I/O.
            # DIO 1 (shutter) and DIO 2 (camera sync) are driven exclusively
            # by the Digital Out pattern generator during a pulse sequence,
            # so they must NOT be claimed here.
            self.ads.digital_io_set_output_enable(0x01)
        except Exception as e:
            print(f"[Warning] Analog Discovery device could not connect: {e}")
            self.ads = None

        # Different Allied Vision models expose different GPIO line names
        # and trigger selector entries. Query what this specific
        # camera actually supports instead of hardcoding a guess.
        self.available_trigger_lines = []
        self.available_trigger_sources = []
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
                self.available_trigger_sources = list(self.camera.list_hardware_trigger_sources())
                print(f"[Camera] Trigger sources reported by this camera: {self.available_trigger_sources}")
            except Exception as e:
                print(f"[Camera] Could not enumerate trigger sources: {e}")

            try:
                self.available_trigger_selectors = [str(entry) for entry in self.camera._cam.TriggerSelector.get_all_entries()]
                print(f"[Camera] Trigger selectors reported by this camera: {self.available_trigger_selectors}")
            except Exception as e:
                print(f"[Camera] Could not enumerate trigger selectors: {e}")
        except Exception as e:
            print(f"[Warning] Allied Vision Camera could not connect: {e}")
            self.camera = None

    def create_four_panels(self):
        """
        Construct a resizable 2x2 layout using nested PanedWindows.

        Using a horizontal PanedWindow per row (rather than a single shared
        grid) lets the two panels in the top row be sized independently
        from the two panels in the bottom row -- e.g. the top-left panel
        can be narrower than the top-right panel, while on the bottom row
        the right panel can be wider than the left one. A single 2x2
        columnconfigure/rowconfigure grid forces both rows to share the
        same column widths, which made it impossible to size any one
        panel independently of the panel directly above/below it. The
        outer vertical PanedWindow also lets the whole top row be shrunk
        relative to the bottom row, and every sash remains user-draggable.
        """
        # A Notebook hosts the original 2x2 layout as its first tab, so the
        # new "PID Lock" tab (Scope Ch2 error -> WaveGen Ch1 modulation)
        # can sit alongside it without disturbing any of the existing
        # panel/pane wiring below -- everything is still parented exactly
        # as before, just one level deeper (inside self.main_tab instead of
        # directly inside self).
        self.main_notebook = ttk.Notebook(self)
        self.main_notebook.pack(fill="both", expand=True, padx=4, pady=4)

        self.main_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.main_tab, text="Main Control")

        self.laser_lock_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.laser_lock_tab, text="Laser Lock (Scope Ch1/Ch2 \u2192 WaveGen Ch1)")
        self.build_laser_lock_panel(self.laser_lock_tab)

        self.main_vertical_pane = ttk.PanedWindow(self.main_tab, orient="vertical")
        self.main_vertical_pane.pack(fill="both", expand=True, padx=4, pady=4)

        self.top_row_pane = ttk.PanedWindow(self.main_vertical_pane, orient="horizontal")
        self.bottom_row_pane = ttk.PanedWindow(self.main_vertical_pane, orient="horizontal")

        # Top row is given less relative height than the bottom row, so
        # both top panels are a little smaller and the bottom row (scope +
        # data extraction controls) gets the extra room it needs.
        self.main_vertical_pane.add(self.top_row_pane, weight=2)
        self.main_vertical_pane.add(self.bottom_row_pane, weight=3)

        # 1. Top Left: Pulse Configuration and Waveform Preview (kept
        # narrower than the top-right camera view -- weight=2 vs weight=3)
        self.p_top_left = ttk.LabelFrame(self.top_row_pane, text="Pulse Control Sequence Settings")
        self.top_row_pane.add(self.p_top_left, weight=2)
        self.build_top_left_panel()

        # 2. Top Right: Live Camera Matrix Control & Parameters
        self.p_top_right = ttk.LabelFrame(self.top_row_pane, text="Camera Interface & Live Video")
        self.top_row_pane.add(self.p_top_right, weight=3)
        self.build_top_right_panel()

        # 3. Bottom Left: Oscilloscope Trace (Fluorescence PD3) & Magnet IO Switches
        self.p_bottom_left = ttk.LabelFrame(self.bottom_row_pane, text="Fluorescence (PD3) Scope & Magnet Control")
        self.bottom_row_pane.add(self.p_bottom_left, weight=2)
        self.build_bottom_left_panel()

        # 4. Bottom Right: Signal Processing Matrix (Snapshot Subtraction
        # Array). Given more relative width (weight=3) than bottom-left so
        # its row of buttons/entries has room to lay out without clipping.
        self.p_bottom_right = ttk.LabelFrame(self.bottom_row_pane, text="Data Extraction & Background Profiles")
        self.bottom_row_pane.add(self.p_bottom_right, weight=3)
        self.build_bottom_right_panel()

    # =========================================================================
    # LASER LOCK TAB (Scope Ch1 = PD2 fluorescence [display only],
    # Ch2 = DAVLL error -> WaveGen Ch1 modulation = sweep + offset +
    # internal bias + PID feedback, summed)
    # =========================================================================

    def build_laser_lock_panel(self, parent):
        """
        Builds the "Laser Lock" tab.

        Signal chain (see _laser_lock_step for the exact implementation):
            Scope Ch2 (DAVLL error) --> (+) internal bias --> PID controller
                --> (+) sweep --> (+) offset --> clamp to +/-5 V
                --> WaveGen Ch1 (laser modulation)

        Scope Ch1 (PD2, plain fluorescence/absorption) is display-only and
        is never fed into the loop. It's read and shown while the lock is
        OFF; once the lock is engaged this tab stops sampling it entirely
        so it's free for the other MOT tab that also needs it.

        This method only builds the controls -- the feedback loop itself
        runs continuously on a background thread started in __init__ (see
        _laser_lock_loop_worker), so sweep/lock state is honored no matter
        which tab is currently visible.
        """
        container = ttk.Frame(parent, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container, text="Laser Lock", font=("Arial", 13, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(
            container,
            text=(
                "Inputs:  Scope Ch1 = PD2 (fluorescence/absorption, display only)\n"
                "         Scope Ch2 = PD1b \u2212 PD1a (DAVLL error signal)\n"
                "Output:  WaveGen Channel 1 = sweep + offset + PID feedback\n"
                "         (feedback acts on Ch2 + internal bias)\n"
                "Output is always hard-limited to \u00b15 V, regardless of gain settings.\n"
                "Engaging the lock turns the sweep OFF and the PID feedback ON."
            ),
            foreground="#666666", justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # --- Live scope / output readouts ---
        readout_frame = ttk.LabelFrame(container, text="Live Scope Traces")
        readout_frame.grid(row=2, column=0, columnspan=2, sticky="we", pady=(0, 12))
        ttk.Label(readout_frame, text="Scope Ch1 (PD2, fluorescence/absorption):").grid(
            row=0, column=0, sticky="w", padx=6, pady=(4, 0)
        )
        self.scope_ch1_canvas = tk.Canvas(readout_frame, bg="#000000", height=140, highlightthickness=0)
        self.scope_ch1_canvas.grid(row=1, column=0, sticky="we", padx=6, pady=(0, 8))
        ttk.Label(readout_frame, text="Scope Ch2 (PD1b \u2212 PD1a, DAVLL error signal):").grid(
            row=2, column=0, sticky="w", padx=6, pady=(4, 0)
        )
        self.scope_ch2_canvas = tk.Canvas(readout_frame, bg="#000000", height=140, highlightthickness=0)
        self.scope_ch2_canvas.grid(row=3, column=0, sticky="we", padx=6, pady=(0, 4))
        readout_frame.columnconfigure(0, weight=1)

        # --- Lock on/off ---
        self.var_lock_enabled = tk.BooleanVar(value=False)
        self.btn_lock_toggle = tk.Button(
            container, text="LOCK: OFF (sweeping)", bg="#5f1e1e", fg="white",
            font=("Arial", 12, "bold"), width=22,
            command=self.toggle_lock_enabled,
        )
        self.btn_lock_toggle.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 14))

        # --- Sweep controls (active while unlocked) ---
        sweep_frame = ttk.LabelFrame(container, text="Sweep (active while unlocked)")
        sweep_frame.grid(row=4, column=0, columnspan=2, sticky="we", pady=(0, 10))
        self.var_sweep_freq_hz = tk.DoubleVar(value=10.0)
        self.var_sweep_amp_v = tk.DoubleVar(value=0.0)
        ttk.Label(sweep_frame, text="Frequency (Hz):").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(sweep_frame, textvariable=self.var_sweep_freq_hz, width=12).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(sweep_frame, text="Amplitude (V, peak):").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(sweep_frame, textvariable=self.var_sweep_amp_v, width=12).grid(row=1, column=1, sticky="w", padx=6, pady=3)

        # --- Offset (slider + numeric entry, mV) ---
        offset_frame = ttk.LabelFrame(container, text="Offset (mV) \u2014 compensates for sweep being off")
        offset_frame.grid(row=5, column=0, columnspan=2, sticky="we", pady=(0, 10))
        self.var_offset_mv = tk.DoubleVar(value=0.0)
        tk.Scale(
            offset_frame, from_=-5000, to=5000, resolution=1, orient="horizontal",
            variable=self.var_offset_mv, length=320, showvalue=False,
        ).grid(row=0, column=0, sticky="we", padx=6, pady=3)
        ttk.Entry(offset_frame, textvariable=self.var_offset_mv, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        offset_frame.columnconfigure(0, weight=1)

        # --- Internal bias (slider + numeric entry, mV) ---
        bias_frame = ttk.LabelFrame(container, text="Internal Bias (mV) \u2014 shifts the PID's lock point, live-adjustable")
        bias_frame.grid(row=6, column=0, columnspan=2, sticky="we", pady=(0, 10))
        self.var_bias_mv = tk.DoubleVar(value=0.0)
        tk.Scale(
            bias_frame, from_=-5000, to=5000, resolution=1, orient="horizontal",
            variable=self.var_bias_mv, length=320, showvalue=False,
        ).grid(row=0, column=0, sticky="we", padx=6, pady=3)
        ttk.Entry(bias_frame, textvariable=self.var_bias_mv, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        bias_frame.columnconfigure(0, weight=1)

        # --- PID controls ---
        pid_frame = ttk.LabelFrame(container, text="PID (acts on Ch2 + bias)")
        pid_frame.grid(row=7, column=0, columnspan=2, sticky="we", pady=(0, 10))

        ttk.Label(pid_frame, text="Polarity:").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self.var_pid_polarity_inverted = tk.BooleanVar(value=False)
        polarity_frame = ttk.Frame(pid_frame)
        polarity_frame.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(
            polarity_frame, text="Normal (+)",
            variable=self.var_pid_polarity_inverted, value=False,
        ).pack(side="left")
        ttk.Radiobutton(
            polarity_frame, text="Inverted (\u2212)",
            variable=self.var_pid_polarity_inverted, value=True,
        ).pack(side="left", padx=(10, 0))

        self.var_pid_g = tk.DoubleVar(value=1.0)
        self.var_pid_p = tk.DoubleVar(value=1.0)
        self.var_pid_i = tk.DoubleVar(value=0.0)
        self.var_pid_d = tk.DoubleVar(value=0.0)

        gain_rows = [
            ("Overall Gain (G):", self.var_pid_g),
            ("Proportional (P):", self.var_pid_p),
            ("Integral (I):", self.var_pid_i),
            ("Derivative (D):", self.var_pid_d),
        ]
        for offset, (label_text, var) in enumerate(gain_rows):
            row = 1 + offset
            ttk.Label(pid_frame, text=label_text).grid(row=row, column=0, sticky="w", padx=6, pady=2)
            ttk.Entry(pid_frame, textvariable=var, width=12).grid(row=row, column=1, sticky="w", padx=6, pady=2)

        # --- Reset integrator (guards against a stale/wound-up integral
        # term left over from a previous lock attempt) ---
        ttk.Button(
            pid_frame, text="Reset Integrator", command=self.reset_pid_integrator,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=(8, 4))

        container.columnconfigure(1, weight=1)

    def toggle_lock_enabled(self):
        """Flips the lock on/off and updates the button's appearance."""
        self.var_lock_enabled.set(not self.var_lock_enabled.get())
        self._refresh_lock_button_appearance()

    def _refresh_lock_button_appearance(self):
        if self.var_lock_enabled.get():
            self.btn_lock_toggle.config(text="LOCK: ON (feedback)", bg="#1e5f2e")
        else:
            self.btn_lock_toggle.config(text="LOCK: OFF (sweeping)", bg="#5f1e1e")

    def reset_pid_integrator(self):
        """Zeroes the integral/derivative history without touching the lock state."""
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = None

    def _laser_lock_arm_wavegen_channel(self):
        """Claims the WaveGen modulation channel and starts it at 0 V."""
        ch = self.WAVEGEN_MOD_CHANNEL
        self.ads.analog_out_reset(ch)
        self.ads.analog_out_enable_node(ch, AnalogOutNodeCarrier, 1)
        self.ads.analog_out_set_function(ch, funcDC)
        self.ads.analog_out_set_offset(ch, 0.0)
        self.ads.analog_out_start(ch)

    def _laser_lock_disarm_wavegen_channel(self):
        """Forces the modulation output back to 0 V and stops the generator."""
        ch = self.WAVEGEN_MOD_CHANNEL
        try:
            self.ads.analog_out_set_offset(ch, 0.0)
            self.ads.analog_out_start(ch)  # push the 0 V update before stopping
            self.ads.analog_out_stop(ch)
        except Exception as e:
            print(f"[Laser Lock] Could not disarm WaveGen channel cleanly: {e}")

    def _set_laser_lock_wavegen_output(self, voltage_v):
        """Updates the live modulation output voltage (already clamped by the caller)."""
        ch = self.WAVEGEN_MOD_CHANNEL
        self.ads.analog_out_set_offset(ch, voltage_v)
        self.ads.analog_out_start(ch)

    def _sweep_value_v(self, now):
        """
        Software-generated symmetric triangle wave, amplitude/frequency
        taken live from the GUI. Uses an absolute time base so changing
        frequency/amplitude on the fly doesn't require any extra state --
        a frequency change just produces a phase discontinuity, the same
        as retuning a hardware function generator.
        """
        freq_hz = self.var_sweep_freq_hz.get()
        amp_v = self.var_sweep_amp_v.get()
        if freq_hz <= 0:
            return 0.0
        phase = (now - self._sweep_t0) * freq_hz
        frac = phase - np.floor(phase)
        tri = 2.0 * abs(2.0 * (frac - np.floor(frac + 0.5))) - 1.0  # triangle in [-1, 1]
        return float(amp_v * tri)

    def _laser_lock_step(self, now, locked):
        """
        Runs a single iteration of the full signal chain and writes the
        result out to WaveGen Ch1:

            Scope Ch2 (error) --> (+) internal bias --> PID
                --> (+) sweep --> (+) offset --> clamp(+/-5V) --> out

        Ch1 (PD2) is only sampled/displayed while unlocked -- see the
        class docstring / build_laser_lock_panel for why.
        """
        dt = (now - self._pid_last_time) if self._pid_last_time is not None else 0.0
        self._pid_last_time = now

        # --- Ch1 (PD2), display-only, skipped entirely while locked ---
        if not locked:
            try:
                ch1_v = self.ads.analog_in_read_sample(channel=self.SCOPE_CH1_FLUOR)
            except Exception:
                ch1_v = None
            self._last_scope_ch1_v = ch1_v
            if ch1_v is not None:
                self._scope_ch1_trace.append((now, ch1_v))
        else:
            self._last_scope_ch1_v = None

        # --- Ch2 (DAVLL error): read, apply polarity ---
        raw_error_v = self.ads.analog_in_read_sample(channel=self.SCOPE_CH2_ERROR)
        self._last_scope_ch2_v = raw_error_v
        self._scope_ch2_trace.append((now, raw_error_v))
        sign = -1.0 if self.var_pid_polarity_inverted.get() else 1.0
        error = sign * raw_error_v

        # --- Internal bias shifts the PID's zero; live-adjustable ---
        bias_v = self.var_bias_mv.get() / 1000.0
        error_biased = error + bias_v

        # --- PID ---
        g_gain = self.var_pid_g.get()
        p_gain = self.var_pid_p.get()
        i_gain = self.var_pid_i.get()
        d_gain = self.var_pid_d.get()

        derivative = ((error_biased - self._pid_last_error) / dt) if dt > 0 else 0.0
        trial_integral = self._pid_integral + (error_biased * dt if dt > 0 else 0.0)
        pid_raw = g_gain * (p_gain * error_biased + i_gain * trial_integral + d_gain * derivative)
        self._pid_last_error = error_biased

        # --- Sum: feedback (locked) or sweep (unlocked), plus offset ---
        offset_v = self.var_offset_mv.get() / 1000.0
        if locked:
            feedback_term = pid_raw
            sweep_term = 0.0
        else:
            feedback_term = 0.0
            sweep_term = self._sweep_value_v(now)

        raw_sum = feedback_term + sweep_term + offset_v
        output = max(-5.0, min(5.0, raw_sum))

        # Simple clamped-integrator anti-windup: only fold the new integral
        # term in (and only while locked) if doing so didn't require
        # clipping the output. This keeps a long-saturated error from
        # leaving behind a huge integral that then overshoots wildly once
        # the loop recovers, and keeps the integrator from accumulating at
        # all while unlocked (it's reset on re-lock anyway, see below).
        if locked and raw_sum == output:
            self._pid_integral = trial_integral

        self._set_laser_lock_wavegen_output(output)

    def _laser_lock_loop_worker(self):
        """
        Background feedback-loop thread. Runs for the entire lifetime of
        the application (started once in __init__, stopped in
        on_app_close), independent of which GUI tab happens to be showing
        -- it only ever reads the tk.Variables built in
        build_laser_lock_panel() and never touches a widget directly, so
        switching tabs has no effect on whether it keeps running.

        Unlike the old PID-only tab, the WaveGen channel is armed as soon
        as the ADS is available and stays armed continuously -- the tab
        sweeps by default (amplitude 0 V until the user dials one in) and
        switches to PID feedback only while locked, rather than sitting
        idle at 0 V until a button is pressed.
        """
        armed = False
        prev_locked = False
        while not self._lock_stop_event.is_set():
            try:
                if self.ads is not None and not armed:
                    self._laser_lock_arm_wavegen_channel()
                    armed = True

                if armed:
                    locked = bool(self.var_lock_enabled.get())

                    if locked and not prev_locked:
                        # Rising edge: clear the integrator so a stale
                        # accumulation from a previous lock attempt
                        # doesn't cause a jump the moment it re-engages.
                        self._pid_integral = 0.0
                        self._pid_last_error = 0.0
                    prev_locked = locked

                    self._laser_lock_step(time.perf_counter(), locked)
            except Exception as e:
                print(f"[Laser Lock Loop Error] {e}")
            time.sleep(self._laser_lock_loop_period_s)

        # Loop is exiting (app shutdown) -- leave the output at a known-
        # safe 0 V rather than abandoning it wherever it last was.
        if armed and self.ads is not None:
            self._laser_lock_disarm_wavegen_channel()

    def _laser_lock_refresh_display(self):
        """
        Low-rate (~7 Hz) GUI refresh for the Scope Ch1/Ch2 rolling-trace
        plots. Runs on the main thread via self.after and only ever reads
        list() snapshots of the deques written by the background loop
        thread -- it never touches self.ads directly, so it can't race
        with that thread.
        """
        if not getattr(self, "_laser_lock_gui_alive", False):
            return
        try:
            ch1_note = "not sampled while locked" if self.var_lock_enabled.get() else None
            self._draw_live_scope_canvas(self.scope_ch1_canvas, list(self._scope_ch1_trace), ch1_note)
            self._draw_live_scope_canvas(self.scope_ch2_canvas, list(self._scope_ch2_trace), None)
        except Exception:
            pass
        self.after(150, self._laser_lock_refresh_display)

    def _draw_live_scope_canvas(self, canvas, data_points, title_note=None):
        """
        Renders a rolling voltage-vs-time trace onto the given tk.Canvas
        from a list of (t, v) tuples (oldest first). Styled the same as
        render_oscilloscope_canvas_trace, adapted for a continuously
        updating rolling buffer instead of a single fixed-length capture.
        """
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10:
            w, h = 360, 140

        LEFT_MARGIN, RIGHT_MARGIN, TOP_MARGIN, BOTTOM_MARGIN = 55, 12, 10, 26
        plot_x0, plot_x1 = LEFT_MARGIN, w - RIGHT_MARGIN
        plot_y0, plot_y1 = TOP_MARGIN, h - BOTTOM_MARGIN
        plot_w, plot_h = plot_x1 - plot_x0, plot_y1 - plot_y0
        if plot_w <= 0 or plot_h <= 0:
            return

        has_data = data_points is not None and len(data_points) >= 2
        if has_data:
            times = [p[0] for p in data_points]
            values = [p[1] for p in data_points]
            t0, t1 = times[0], times[-1]
            span_t = (t1 - t0) if (t1 - t0) > 1e-6 else 1.0
            v_min, v_max = min(values), max(values)
        else:
            t0, span_t = 0.0, 1.0
            v_min, v_max = 0.0, 1.0
        span_v = (v_max - v_min) if (v_max - v_min) > 0.01 else 1.0

        mid_y = plot_y0 + plot_h / 2
        canvas.create_line(plot_x0, mid_y, plot_x1, mid_y, fill="#222222")

        # --- Y axis (voltage) ---
        canvas.create_line(plot_x0, plot_y0, plot_x0, plot_y1, fill="#555555", width=1)
        n_y_ticks = 4
        for i in range(n_y_ticks + 1):
            frac = i / n_y_ticks
            y_val = v_max - frac * span_v
            y_pix = plot_y0 + frac * plot_h
            canvas.create_line(plot_x0 - 4, y_pix, plot_x0, y_pix, fill="#555555")
            canvas.create_text(
                plot_x0 - 6, y_pix, text=f"{y_val:.3f}V", fill="#888888",
                font=("Arial", 8), anchor="e",
            )
        canvas.create_text(12, plot_y0 - 2, text="V", fill="#888888", font=("Arial", 8), anchor="nw")

        # --- X axis (seconds ago, 0 = now) ---
        canvas.create_line(plot_x0, plot_y1, plot_x1, plot_y1, fill="#555555", width=1)
        n_x_ticks = 4
        for i in range(n_x_ticks + 1):
            frac = i / n_x_ticks
            x_pix = plot_x0 + frac * plot_w
            canvas.create_line(x_pix, plot_y1, x_pix, plot_y1 + 4, fill="#555555")
            t_val = -(1.0 - frac) * span_t
            anchor = "n" if i not in (0, n_x_ticks) else ("nw" if i == 0 else "ne")
            canvas.create_text(
                x_pix, plot_y1 + 6, text=f"{t_val:.2f}s", fill="#888888",
                font=("Arial", 8), anchor=anchor,
            )
        canvas.create_text(plot_x1, h - 4, text="Time", fill="#888888", font=("Arial", 8), anchor="se")

        if title_note:
            canvas.create_text(
                plot_x0 + 4, plot_y0 + 2, text=title_note, fill="#ffaa00",
                font=("Arial", 8, "italic"), anchor="nw",
            )

        if not has_data:
            return

        points = []
        for (t, v) in data_points:
            x_pixel = plot_x0 + ((t - t0) / span_t) * plot_w
            y_pixel = plot_y1 - ((v - v_min) / span_v) * plot_h
            points.append((x_pixel, y_pixel))
        flat_points = [coord for pt in points for coord in pt]
        canvas.create_line(flat_points, fill="#00ff00", width=1.2)

        canvas.create_text(plot_x1 - 4, plot_y0 + 10, text=f"Max: {v_max:.3f} V", fill="#888888", font=("Arial", 8), anchor="e")
        canvas.create_text(plot_x1 - 4, plot_y1 - 10, text=f"Min: {v_min:.3f} V", fill="#888888", font=("Arial", 8), anchor="e")

    # =========================================================================
    # PANEL BUILDERS & LOGIC SECTIONS
    # =========================================================================

    def build_top_left_panel(self):
        """User parameter dashboard specifying sequencing configurations."""
        container = ttk.Frame(self.p_top_left, padding=6)
        container.pack(fill="both", expand=True)

        # Config Variables
        self.var_time_after_pulse = tk.DoubleVar(value=5.0)  # ms
        self.var_time_between_pulses = tk.DoubleVar(value=1.0) # s
        self.var_pd3_window = tk.DoubleVar(value=50.0)       # ms
        self.var_num_pulses = tk.IntVar(value=1)

        # Form Controls layout grid
        lbl_style = {"sticky": "w", "padx": 4, "pady": 1}
        ttk.Label(container, text="Time after pulse to snap (ms):").grid(row=0, column=0, **lbl_style)
        ttk.Entry(container, textvariable=self.var_time_after_pulse, width=9).grid(row=0, column=1, sticky="w")

        ttk.Label(container, text="Time between repeat pulses (s):").grid(row=1, column=0, **lbl_style)
        ttk.Entry(container, textvariable=self.var_time_between_pulses, width=9).grid(row=1, column=1, sticky="w")

        ttk.Label(container, text="Fluorescence PD3 Domain (ms):").grid(row=2, column=0, **lbl_style)
        ttk.Entry(container, textvariable=self.var_pd3_window, width=9).grid(row=2, column=1, sticky="w")

        ttk.Label(container, text="Number of Pulses:").grid(row=3, column=0, **lbl_style)
        ttk.Entry(container, textvariable=self.var_num_pulses, width=9).grid(row=3, column=1, sticky="w")

        # Synchronize-magnet checkbox lives next to the pulse parameters now
        # (moved here from the bottom-left Magnet Control panel) since it
        # directly governs how this sequence preview and the pulse train
        # itself treat the magnet (DIO 0) channel.
        self.var_sync_pulse = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            container, text="Synchronize Magnet Line with Pulse Sequence", variable=self.var_sync_pulse
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 1))

        # Visual Plot Canvas for Matrix Preview Strategy
        ttk.Label(container, text="Intended Signal Trajectory Preview:").grid(row=5, column=0, columnspan=2, sticky="w", pady=(8,2))
        self.sequence_canvas = tk.Canvas(container, height=175, bg="#1e1e1e", highlightthickness=0)
        self.sequence_canvas.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=4)
        container.rowconfigure(6, weight=1)
        container.columnconfigure(1, weight=1)
        # Re-draw the preview whenever the canvas is resized, so the time
        # axis and its labels are never clipped at the current width.
        self.sequence_canvas.bind("<Configure>", lambda evt: self.render_sequence_preview_graph())

        # Pulse trigger button (moved here from the top-right camera panel,
        # since firing a pulse is fundamentally a sequence-timing action).
        self.btn_synch_pulse = tk.Button(
            container, text="PULSE", bg="#2f2525", fg="white",
            font=("Arial", 11, "bold"), command=self.execute_synch_pulse_routine
        )
        self.btn_synch_pulse.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(4, 6))

        # Auto-save pulse image arrays + base filename (moved here from the
        # bottom-right Data Extraction panel).
        self.var_auto_save_pulsed_img = tk.BooleanVar(value=False)
        self.var_pulse_filename_base = tk.StringVar(value="pulsed_frame_capture")

        autosave_row = ttk.Frame(container)
        autosave_row.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        autosave_row.columnconfigure(2, weight=1)

        ttk.Checkbutton(
            autosave_row, text="Auto-Save Pulse Image Arrays", variable=self.var_auto_save_pulsed_img
        ).grid(row=0, column=0, padx=(4, 8), sticky="w")
        ttk.Label(autosave_row, text="Base Filename:").grid(row=0, column=1, sticky="w")
        ttk.Entry(autosave_row, textvariable=self.var_pulse_filename_base).grid(row=0, column=2, padx=4, sticky="ew")
        ttk.Button(
            autosave_row, text="Browse", command=self.browse_pulse_filename_base
        ).grid(row=0, column=3, padx=(4, 0))

        # Re-draw the visual timing preview trace every time values shift
        for var in [self.var_time_after_pulse, self.var_time_between_pulses, self.var_pd3_window]:
            var.trace_add("write", lambda *args: self.render_sequence_preview_graph())
        self.var_num_pulses.trace_add("write", lambda *args: self.render_sequence_preview_graph())
        # The magnet channel's rendering depends on whether it's synced to
        # the pulse train, and (when it isn't) on its current static value.
        self.var_sync_pulse.trace_add("write", lambda *args: self.render_sequence_preview_graph())

        self.render_sequence_preview_graph()

    def render_sequence_preview_graph(self):
        """Draws a multi-channel timing schematic with a real time axis and repeated pulses."""
        self.sequence_canvas.delete("all")
        w = self.sequence_canvas.winfo_width() if self.sequence_canvas.winfo_width() > 50 else 500
        h = self.sequence_canvas.winfo_height() if self.sequence_canvas.winfo_height() > 50 else 175

        try:
            snap_delay_ms   = self.var_time_after_pulse.get()
            pd3_domain_ms   = self.var_pd3_window.get()
            between_s       = self.var_time_between_pulses.get()
            num_pulses      = max(1, self.var_num_pulses.get())
        except Exception:
            return  # suppress entry-parsing hiccups during typing

        # Magnet (DIO 0) is only part of the pulse train when the
        # "Synchronize" checkbox is on; otherwise it just sits at whatever
        # static level the manual toggle in the Magnet Control panel is
        # set to, and is rendered as a flat DC line instead of a pulse.
        sync_magnet   = bool(self.var_sync_pulse.get())
        magnet_is_high = bool(getattr(self, "var_magnet_state", tk.BooleanVar(value=False)).get())

        # Total time window to display (ms)
        between_ms   = between_s * 1000.0
        total_ms     = (pd3_domain_ms + between_ms) * num_pulses
        if total_ms <= 0:
            return

        # Layout constants. RIGHT_MARGIN is generous enough to fit the
        # right-most time-axis tick label (e.g. "1000ms") without it being
        # clipped by the edge of the canvas -- with anchor="n" the label is
        # centered on its tick, so roughly half its width extends to the
        # right of the last tick mark. TOP_MARGIN leaves headroom above the
        # plot for the "PD3" region label, and AXIS_HEIGHT leaves enough
        # room below the plot for the time-axis tick marks and their text
        # labels -- both were previously tall enough to get clipped by the
        # canvas edge.
        LEFT_MARGIN  = 72   # px for channel labels
        RIGHT_MARGIN = 40
        TOP_MARGIN   = 22
        AXIS_HEIGHT  = 26   # px for the time axis at the bottom
        plot_w = w - LEFT_MARGIN - RIGHT_MARGIN
        plot_h = h - TOP_MARGIN - AXIS_HEIGHT
        if plot_w <= 0 or plot_h <= 0:
            return

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

        # Dashed vertical markers showing the PD3 fluorescence measurement
        # domain/region for every pulse -- i.e. the window from each
        # pulse's rising edge to pd3_domain_ms later.
        plot_top = TOP_MARGIN
        plot_bottom = TOP_MARGIN + plot_h
        t = 0.0
        pd3_label_drawn = False
        for p in range(num_pulses):
            t_start = t
            t_end = t + pd3_domain_ms
            x_start = t2x(t_start)
            x_end = t2x(min(t_end, total_ms))
            for x_mark in (x_start, x_end):
                self.sequence_canvas.create_line(
                    x_mark, plot_top, x_mark, plot_bottom,
                    fill="#ff5252", dash=(4, 3), width=1
                )
            if not pd3_label_drawn:
                self.sequence_canvas.create_text(
                    (x_start + x_end) / 2, plot_top - 2,
                    text="PD3", fill="#ff5252", font=("Consolas", 7), anchor="s"
                )
                pd3_label_drawn = True
            t = t + pd3_domain_ms + between_ms

        # Draw pulses for each channel
        for row, (label, color, pin) in enumerate(channels):
            base_y = row_y(row)
            high_y = base_y - int(ch_h * 0.75)

            # Magnet channel, when not synchronized to the pulse train,
            # just displays its current static value as a flat line.
            if pin == 0 and not sync_magnet:
                level_y = high_y if magnet_is_high else base_y
                self.sequence_canvas.create_line(
                    t2x(0), level_y, t2x(total_ms), level_y, fill=color, width=1
                )
                state_txt = "STATIC HIGH" if magnet_is_high else "STATIC LOW"
                self.sequence_canvas.create_text(
                    t2x(total_ms) - 4, level_y - 6, text=state_txt,
                    fill=color, font=("Consolas", 7), anchor="e"
                )
                continue

            # The camera sync pin shares the SAME rising edge as the magnet
            # and shutter channels, but its high-time is "time after pulse
            # to snap" rather than the full PD3 domain -- its falling edge
            # is what fires the camera's hardware trigger, so the falling
            # edge lands exactly at the intended snap time.
            chan_high_ms = snap_delay_ms if pin == 2 else pd3_domain_ms

            t = 0.0
            x0 = t2x(0)
            for p in range(num_pulses):
                x_rise = t2x(t)
                t_fall = t + chan_high_ms
                x_fall = t2x(min(t_fall, total_ms))
                t_next = t + pd3_domain_ms + between_ms
                x_next = t2x(min(t_next, total_ms))

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
            # Anchor the final tick's label so it can't run past the right
            # edge of the canvas, even with the widened RIGHT_MARGIN.
            is_last_tick = t_tick + tick_step_ms > total_ms + tick_step_ms * 0.01
            anchor = "ne" if is_last_tick else "n"
            self.sequence_canvas.create_text(
                xt, axis_y + 10, text=lbl, fill="#666666", font=("Consolas", 7), anchor=anchor
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
        # (queried in init_hardware_connections). Our camera should use Line6 and
        # AcquisitionStart to trigger.
        line_options = self.available_trigger_lines or ["Line6"]
        source_options = self.available_trigger_sources or ["InputLines"]
        selector_options = self.available_trigger_selectors or ["AcquisitionStart"]

        self.var_trigger_line = tk.StringVar(value=line_options[0])
        self.var_trigger_source = tk.StringVar(value=source_options[0])
        self.var_trigger_selector = tk.StringVar(value=selector_options[0])

        trig_box = ttk.LabelFrame(controls_frame, text="Hardware Trigger Wiring")
        trig_box.pack(fill="x", pady=4, padx=2)

        ttk.Label(trig_box, text="GPIO Line:").grid(row=0, column=0, **grid_params)
        ttk.Combobox(trig_box, textvariable=self.var_trigger_line, values=line_options, width=14, state="readonly").grid(row=0, column=1, sticky="w")
        
        ttk.Label(trig_box, text="Trigger Source:").grid(row=1, column=0, **grid_params)
        ttk.Combobox(trig_box, textvariable=self.var_trigger_source, values=source_options, width=14, state="readonly").grid(row=1, column=1, sticky="w")

        ttk.Label(trig_box, text="Trigger Selector:").grid(row=2, column=0, **grid_params)
        ttk.Combobox(trig_box, textvariable=self.var_trigger_selector, values=selector_options, width=14, state="readonly").grid(row=2, column=1, sticky="w")

        if not self.available_trigger_lines or not self.available_trigger_selectors or not self.available_trigger_sources:
            ttk.Label(trig_box, text="(No camera connected -- placeholder values shown)", foreground="#cc8800").grid(row=2, column=0, columnspan=2, sticky="w", padx=2)

        # Operational Control Buttons Pack
        action_box = ttk.LabelFrame(controls_frame, text="Direct Trigger Operations")
        action_box.pack(fill="x", pady=4, padx=2)

        ttk.Button(action_box, text="Capture Snapshot Now", command=self.execute_immediate_snapshot).pack(fill="x", pady=2)
        ttk.Button(action_box, text="Extract & Save Background", command=self.capture_background_profile).pack(fill="x", pady=2)
        # NOTE: the pulse-trigger button now lives in the top-left panel
        # (self.btn_synch_pulse is created in build_top_left_panel), next
        # to the sequence-timing parameters it actually fires.

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
        controls_sub.columnconfigure(0, weight=1)

        # Magnet toggle switch. NOTE: the "Synchronize Line with Pulse
        # Sequence" checkbox that used to sit next to this has moved to
        # the top-left panel, alongside the pulse-timing parameters it
        # actually governs.
        self.var_magnet_state = tk.BooleanVar(value=False)
        self.chk_magnet = ttk.Checkbutton(controls_sub, text="Enable Magnet Power (Static DIO 0)", variable=self.var_magnet_state, command=self.toggle_magnet_static_line)
        self.chk_magnet.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 2))
        # Also drives the top-left preview graph's flat "static value" line
        # for the magnet channel whenever sync is off.
        self.var_magnet_state.trace_add("write", lambda *args: self.render_sequence_preview_graph())

        # CSV Logging Parameter Widgets
        self.var_save_csv = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_sub, text="Save 'Fluorescence (PD3)' Trace", variable=self.var_save_csv).grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))

        # The path entry + browse button get their own full-width row so
        # the entry can shrink/grow with the panel while the button always
        # keeps its natural size -- previously both shared row 1 of a
        # 3-column grid with no expanding column, so at narrower panel
        # widths the "Browse Destination" button ran past the edge of the
        # panel and was cut off.
        path_row = ttk.Frame(controls_sub)
        path_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 4))
        path_row.columnconfigure(0, weight=1)

        self.var_csv_path = tk.StringVar(value=os.path.join(os.getcwd(), "fluorescence_output.csv"))
        ttk.Entry(path_row, textvariable=self.var_csv_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(path_row, text="Browse Destination", command=self.browse_csv_destination_file).grid(row=0, column=1, padx=(6, 0))

        # WaveForms Oscilloscope Trace Visual Canvas Display Component
        ttk.Label(container, text="Oscilloscope Buffer Display: Channel 0 (Fluorescence Array Data)").pack(anchor="w", pady=(6,0))
        
        # FIX: Placed inside a dedicated canvas master frame with pack parameters configured to scale properly
        self.scope_canvas = tk.Canvas(container, bg="#000000", height=180, highlightthickness=0)
        self.scope_canvas.pack(fill="both", expand=True, pady=4)

        # Keep the most recent trace so the axes/labels can be redrawn
        # cleanly if the panel is resized (or before any trace exists,
        # in which case just the empty axes are shown).
        self._last_scope_voltages = np.array([])
        self._last_scope_duration_s = None
        self.scope_canvas.bind("<Configure>", lambda evt: self.render_oscilloscope_canvas_trace(
            self._last_scope_voltages, self._last_scope_duration_s
        ))

    def build_bottom_right_panel(self):
        """Maintains image subtraction data arrays, displaying original and processed streams side by side."""
        container = ttk.Frame(self.p_bottom_right, padding=6)
        container.pack(fill="both", expand=True)

        # NOTE: the "Auto-Save Pulse Image Arrays" checkbox and base
        # filename entry have moved to the top-left panel
        # (self.var_auto_save_pulsed_img / self.var_pulse_filename_base are
        # created in build_top_left_panel), next to the Pulse button that
        # triggers the capture they control.
        top_config = ttk.Frame(container)
        top_config.pack(side="top", fill="x", pady=2)

        ttk.Button(top_config, text="Manually Save Current Snapshot", command=self.save_current_snapshot_manually).grid(row=0, column=0, padx=(4, 10), sticky="w")
        ttk.Button(top_config, text="Clear Subtraction Background", command=self.clear_background_buffer).grid(row=0, column=1, padx=4, sticky="w")

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

    def _resolve_trigger_selector(self):
        """
        Map the GUI's selected trigger-selector string back onto the
        TriggerSelector enum expected by HardwareTriggerConfig.

        Different camera models can report GenICam selector names that
        don't line up with one of the enum's known values -- fall back to
        FRAME_START (universally supported by Allied Vision cameras)
        rather than raising in the middle of a pulse sequence.
        """
        try:
            return TriggerSelector(self.var_trigger_selector.get())
        except ValueError:
            print(f"[Camera] Unknown trigger selector '{self.var_trigger_selector.get()}'; "
                  f"defaulting to FrameStart.")
            return TriggerSelector.FRAME_START

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
                # Ensure DIO 0 is still configured as an output (a previous
                # digital_io_reset() call could clear the output-enable mask).
                # Only claim pin 0 here -- pins 1 and 2 belong to the Digital
                # Out pattern generator and must stay unclaimed by Digital
                # I/O, or the pattern generator's pulses get silently
                # overridden by Digital I/O's static (low) drive on those pins.
                self.ads.digital_io_set_output_enable(0x01)
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

    def browse_pulse_filename_base(self):
        """
        Launches a save dialog so the user can point auto-saved pulse
        snapshots at a different folder (and/or change the base filename).

        var_pulse_filename_base is a *base* path, not a full filename --
        finalize_and_render_pulse_metrics() appends "_{timestamp}.png" to
        it for every saved frame. So rather than asking for a single file,
        this seeds the dialog with the current folder/name and strips
        whatever extension the OS dialog appends back off the result,
        leaving a bare base path (which may include a directory) that the
        timestamp + extension get appended to later.
        """
        current = self.var_pulse_filename_base.get()
        initial_dir = os.path.dirname(current) or os.getcwd()
        initial_file = os.path.basename(current) or "pulsed_frame_capture"
        file_path = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension="",
            filetypes=[("All files", "*.*")],
            title="Choose folder + base filename for auto-saved pulse snapshots",
        )
        if file_path:
            base, _ext = os.path.splitext(file_path)
            self.var_pulse_filename_base.set(base)

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
        # calling create_image() on every frame.
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

        # Mark the sequence as in-flight so on_app_close() knows not to
        # close self.ads / self.camera out from under the worker thread
        # below (which will still be actively using both).
        self._sequence_running = True

        # take_snapshot()/arm_hardware_trigger() both raise if continuous
        # streaming is still active, so stop it here on the main thread
        # before the worker thread below starts touching the camera.
        self._stop_camera_live_view()

        def sequence_execution_worker():
            try:
                # 1. Gather all GUI operational parameters safely
                #
                # `delay_to_snap_*` (GUI label "Time after pulse to snap")
                # directly sets the camera (DIO2) pulse's high-time. DIO2
                # shares its RISING edge with the magnet/shutter channels,
                # but has a shorter high-time so its FALLING edge -- which
                # is what the camera is hardware-triggered on -- lands
                # exactly "time after pulse to snap" after the pulse train
                # starts.
                delay_to_snap_ms = self.var_time_after_pulse.get()
                pd3_domain_ms = self.var_pd3_window.get()
                total_pulses = self.var_num_pulses.get()
                
                # Math translations scaling parameters to seconds
                delay_to_snap_s = delay_to_snap_ms / 1000.0
                pd3_domain_s = pd3_domain_ms / 1000.0
                
                print(f"[Pulse Engine] Commencing {total_pulses} synchronized hardware triggers...")

                # Scope buffer size -- computed early so the variable is always
                # in scope for the data-read block at the bottom.
                scope_sample_rate    = 100000.0  # 100 kHz
                scope_buffer_samples = max(1024, int(scope_sample_rate * pd3_domain_s))

                # 2. Program and fire the Digital Out pattern generator.
                #
                # Pin roles:
                #   DIO 0 – magnet coil  (only when "Synchronize" checkbox is on)
                #   DIO 1 – shutter
                #   DIO 2 – camera sync
                pulse_done_event = threading.Event()
                pulse_error      = [None]

                # Whether the camera was successfully armed for a
                # hardware-triggered capture below. Hardware-triggered
                # capture is driven entirely by the DIO2 camera-sync pulse
                # coming out of the ADS pattern generator, so without an
                # ADS connection there is no edge to trigger on.
                camera_hw_armed = False
                if self.camera and self.camera._cam and not self.ads:
                    print("[Pulse Engine Camera] No ADS connected -- skipping hardware-triggered "
                          "capture (the DIO2 camera-sync pulse never fires without the ADS pulse train).")

                if self.ads:
                    between_s = self.var_time_between_pulses.get()
                    # All channels share the same repeat period
                    # (pd3_domain_s + between_s) so they stay in phase
                    # across repeated pulses. The camera (DIO2) pin gets a
                    # shorter high-time -- delay_to_snap_s instead of the
                    # full pd3_domain_s -- with its low-time padded out so
                    # its period still matches every other channel.
                    period_s = pd3_domain_s + between_s
                    cam_high_s = max(0.0, min(delay_to_snap_s, period_s))
                    cam_low_s = period_s - cam_high_s

                    pulse_pins  = [1, 2]
                    pulse_highs = [pd3_domain_s, cam_high_s]
                    pulse_lows  = [between_s, cam_low_s]

                    if self.var_sync_pulse.get():
                        pulse_pins.insert(0, 0)
                        pulse_highs.insert(0, pd3_domain_s)
                        pulse_lows.insert(0, between_s)
                        # Digital I/O normally owns pin 0 (the magnet's static
                        # level). Release its claim so the Digital Out pattern
                        # generator can actually drive it during the pulse --
                        # otherwise Digital I/O's static low/high level wins
                        # and the pulse on pin 0 is invisible, same as the
                        # pin 1/2 issue.
                        try:
                            self.ads.digital_io_set_output_enable(0x00)
                        except Exception as _e:
                            print(f"[Pulse Engine] Could not release DIO0 from Digital I/O: {_e}")

                    total_run_s   = (pd3_domain_s + between_s) * total_pulses
                    pulse_timeout = total_run_s + 2.0  # safety margin

                    print(f"[Pulse Engine] Programming digital_out: pins={pulse_pins} "
                          f"high={pd3_domain_s*1000:.3f}ms low={between_s*1000:.1f}ms "
                          f"x{total_pulses} (run={total_run_s:.4f}s)")

                    # Sanity-check the timing against the device's internal
                    # clock BEFORE firing, so a degenerate (effectively-zero)
                    # pulse width shows up in the console instead of just
                    # silently producing no visible edge.
                    try:
                        clk = self.ads.digital_out_get_internal_clock()
                        for chan_name, chan_high_s in (("shutter/magnet", pd3_domain_s), ("camera", cam_high_s)):
                            min_high_ticks = round(clk * chan_high_s)
                            if min_high_ticks < 1:
                                print(f"[Pulse Engine][WARNING] {chan_name} high_time={chan_high_s*1e6:.1f}us "
                                      f"rounds to <1 tick at clk={clk/1e6:.1f}MHz -- pulse will be "
                                      f"effectively zero width and may not be visible.")
                    except Exception:
                        pass

                    # Arm the camera's hardware trigger BEFORE firing the
                    # pulse train, so it is already waiting in hardware when
                    # the DIO2 camera-sync pulse arrives. DIO2 rises at the
                    # same instant as the shutter/magnet channels, but
                    # falls back low after only cam_high_s (== "time after
                    # pulse to snap"). Triggering on the FALLING edge means
                    # exposure starts the instant DIO2 drops low again --
                    # i.e. exactly delay_to_snap_s after the pulse train
                    # starts, regardless of how long the PD3 domain itself
                    # runs (compare to the old approach, which grabbed
                    # whatever frame happened to be sitting in the live
                    # view buffer with no real timing relationship to the
                    # pulse at all).
                    if self.camera and self.camera._cam:
                        try:
                            hw_cfg = HardwareTriggerConfig(
                                line=self.var_trigger_line.get(),
                                source=self.var_trigger_source.get(),
                                selector=self._resolve_trigger_selector(),
                                activation=TriggerActivation.FALLING_EDGE,
                                acquisition_mode=AcquisitionMode.SINGLE_FRAME,
                                timeout_s=pulse_timeout,
                            )
                            self.camera.arm_hardware_trigger(hw_cfg)
                            camera_hw_armed = True
                            print(f"[Pulse Engine Camera] Armed hardware trigger on "
                                  f"{hw_cfg.line} ({hw_cfg.activation.value}), "
                                  f"selector={hw_cfg.selector.value}.")
                        except Exception as err:
                            print(f"[Pulse Engine Camera] Could not arm hardware trigger: {err}")

                    def _fire_pulse_train():
                        try:
                            self.ads.digital_out_pulse_train(
                                pins=pulse_pins,
                                high_times_s=pulse_highs,
                                low_times_s=pulse_lows,
                                idle_states=DwfDigitalOutIdleLow,
                                pulse_count=total_pulses,
                                wait_for_done=True,
                                timeout_s=pulse_timeout,
                            )
                            print(f"[Pulse Engine] digital_out Done "
                                  f"(status={self.ads.digital_out_status()}).")
                        except Exception as _e:
                            pulse_error[0] = _e
                            print(f"[Pulse Engine] digital_out_pulse_train error: {_e}")
                        finally:
                            pulse_done_event.set()

                    threading.Thread(target=_fire_pulse_train, daemon=True).start()
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
                    self.ads.analog_in_set_trigger_source(trigsrcDigitalOut)
                    self.ads.analog_in_set_trigger_type(0)       # edge
                    self.ads.analog_in_set_trigger_condition(0)  # rising
                    self.ads.analog_in_set_trigger_position(0.5*scope_buffer_samples/scope_sample_rate) # put trigger at start of buffer
                    self.ads.analog_in_configure(reconfigure=True, start=True)
                    print(f"[Pulse Engine] Oscilloscope armed: {scope_buffer_samples} "
                          f"samples @ {scope_sample_rate/1e3:.0f} kHz")

                # 4. Block until the camera's hardware trigger fires (DIO2
                # falling edge) and the triggered frame arrives. The camera
                # was already armed and waiting in hardware above, so this
                # just picks up the frame -- no delay to coordinate against
                # the pulse train, since the falling edge IS the trigger.
                captured_frame = None
                if camera_hw_armed:
                    try:
                        captured_frame = self.camera.wait_for_hardware_trigger(timeout_s=pulse_timeout)
                    except Exception as err:
                        print(f"[Pulse Engine Camera] Hardware-triggered capture failed: {err}")
                    finally:
                        try:
                            self.camera.disarm_hardware_trigger()
                        except Exception as err:
                            print(f"[Pulse Engine Camera] Could not disarm hardware trigger: {err}")

                # 5. Wait for the pulse train to complete, then read scope data.
                _wait_timeout = (total_run_s + 2.0) if self.ads else 1.0
                pulse_done_event.wait(timeout=_wait_timeout)
                if pulse_error[0]:
                    print(f"[Pulse Engine] Pulse error: {pulse_error[0]}")

                # After Digital Out finishes, hand DIO 0 (magnet) back to
                # Digital I/O and restore its static level -- the pulse
                # temporarily took ownership of that pin away from Digital
                # I/O so it could toggle during the sync burst.
                if self.ads and self.var_sync_pulse.get():
                    try:
                        self.ads.digital_io_set_output_enable(0x01)
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
                            print(f"Scope status: {status}") #debug line - currently armed not trig'd
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
            self._last_scope_voltages = scope_voltages
            self._last_scope_duration_s = total_duration_s
            self.render_oscilloscope_canvas_trace(scope_voltages, total_duration_s)
            
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

    def render_oscilloscope_canvas_trace(self, voltage_array, total_duration_s=None):
        """Draws the oscilloscope trace onto the canvas, with labeled X (time) and Y (voltage) axes."""
        self.scope_canvas.delete("all")
        w = self.scope_canvas.winfo_width()
        h = self.scope_canvas.winfo_height()
        if w < 10 or h < 10:
            w, h = 500, 180

        # Axis layout constants
        LEFT_MARGIN   = 55   # room for voltage tick labels
        RIGHT_MARGIN  = 12
        TOP_MARGIN    = 10
        BOTTOM_MARGIN = 26   # room for time tick labels
        plot_x0 = LEFT_MARGIN
        plot_x1 = w - RIGHT_MARGIN
        plot_y0 = TOP_MARGIN
        plot_y1 = h - BOTTOM_MARGIN
        plot_w  = plot_x1 - plot_x0
        plot_h  = plot_y1 - plot_y0

        if plot_w <= 0 or plot_h <= 0:
            return

        has_data = voltage_array is not None and len(voltage_array) >= 2
        if has_data:
            v_min, v_max = float(np.min(voltage_array)), float(np.max(voltage_array))
        else:
            v_min, v_max = 0.0, 1.0
        span = (v_max - v_min) if (v_max - v_min) > 0.01 else 1.0

        # Mid-scale grid reference line
        mid_y = plot_y0 + plot_h / 2
        self.scope_canvas.create_line(plot_x0, mid_y, plot_x1, mid_y, fill="#222222")

        # --- Y axis (voltage) ---
        self.scope_canvas.create_line(plot_x0, plot_y0, plot_x0, plot_y1, fill="#555555", width=1)
        n_y_ticks = 4
        for i in range(n_y_ticks + 1):
            frac = i / n_y_ticks
            y_val = v_max - frac * span
            y_pix = plot_y0 + frac * plot_h
            self.scope_canvas.create_line(plot_x0 - 4, y_pix, plot_x0, y_pix, fill="#555555")
            self.scope_canvas.create_text(
                plot_x0 - 6, y_pix, text=f"{y_val:.3f}V", fill="#888888",
                font=("Arial", 8), anchor="e"
            )
        self.scope_canvas.create_text(
            12, plot_y0 - 2, text="V", fill="#888888", font=("Arial", 8), anchor="nw"
        )

        # --- X axis (time) ---
        self.scope_canvas.create_line(plot_x0, plot_y1, plot_x1, plot_y1, fill="#555555", width=1)
        n_x_ticks = 5
        duration_ms = (total_duration_s * 1000.0) if total_duration_s else None
        for i in range(n_x_ticks + 1):
            frac = i / n_x_ticks
            x_pix = plot_x0 + frac * plot_w
            self.scope_canvas.create_line(x_pix, plot_y1, x_pix, plot_y1 + 4, fill="#555555")
            if duration_ms is not None:
                t_val = frac * duration_ms
                lbl = f"{t_val:.1f}ms"
            else:
                # No timing context available (e.g. before the first trace)
                # -- fall back to a fraction-of-buffer label.
                lbl = f"{frac:.1f}"
            anchor = "n" if i not in (0, n_x_ticks) else ("nw" if i == 0 else "ne")
            self.scope_canvas.create_text(
                x_pix, plot_y1 + 6, text=lbl, fill="#888888", font=("Arial", 8), anchor=anchor
            )
        self.scope_canvas.create_text(
            plot_x1, h - 4, text="Time", fill="#888888", font=("Arial", 8), anchor="se"
        )

        if not has_data:
            return

        # Normalize array values into geometric visual heights within the plot box
        points = []
        for idx, val in enumerate(voltage_array):
            x_pixel = plot_x0 + (idx / (len(voltage_array) - 1)) * plot_w
            y_pixel = plot_y1 - ((val - v_min) / span) * plot_h
            points.append((x_pixel, y_pixel))

        # Flatten point pairs and render a smooth line sequence
        flat_points = [coord for pt in points for coord in pt]
        self.scope_canvas.create_line(flat_points, fill="#00ff00", width=1.5)

        # Add tracking labels to the visual scale bounds
        self.scope_canvas.create_text(plot_x1 - 4, plot_y0 + 10, text=f"Max: {v_max:.3f} V", fill="#888888", font=("Arial", 8), anchor="e")
        self.scope_canvas.create_text(plot_x1 - 4, plot_y1 - 10, text=f"Min: {v_min:.3f} V", fill="#888888", font=("Arial", 8), anchor="e")

    def reset_interface_execution_safeguards(self):
        """Re-enables the GUI inputs and resumes the live video processing loop safely."""
        self.btn_synch_pulse.config(text="PULSE", state="normal")
        self.live_view_active = True
        # The worker thread is done touching self.ads / self.camera now,
        # so it's safe for on_app_close() to proceed with hardware
        # teardown if a close request was left waiting on this.
        self._sequence_running = False
        self._start_camera_live_view()

if __name__ == "__main__":
    app = CoreInstrumentApplication()
    app.mainloop()