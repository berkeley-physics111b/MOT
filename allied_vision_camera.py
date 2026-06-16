"""
Allied Vision Camera Interface (VmbPy / Vimba X)
=================================================
Provides:
  - Camera open / close lifecycle
  - Continuous live-view streaming (display, non-critical timing)
  - Software-triggered snapshot with minimal latency
  - Hardware / external trigger: arm, wait-for-frame, disarm
  - Region-of-interest, gain, brightness, and exposure-time control
  - Pixel-format selection
  - Warning / error logging via the Vimba X log system
  - Camera connection / disconnection callbacks
  - Temperature monitoring (if supported by camera)
  - Settings save / load (XML)

Requirements:
  pip install vmbpy[numpy,opencv]   (whl from the Vimba X SDK directory)
  pip install opencv-python numpy

Hardware trigger quick-start
-----------------------------
  htrig = HardwareTriggerConfig(
      line="Line1",                        # GPIO input line on the camera
      activation=TriggerActivation.RISING_EDGE,
      selector=TriggerSelector.FRAME_START,
      acquisition_mode=AcquisitionMode.SINGLE_FRAME,
  )
  with AlliedVisionCamera(config) as cam:
      cam.arm_hardware_trigger(htrig, callback=my_fn)
      # … external pulse arrives …
      frame = cam.wait_for_hardware_trigger(timeout_s=10.0)
      cam.disarm_hardware_trigger()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from vmbpy import (
    Camera,
    Frame,
    FrameStatus,
    LOG_CONFIG_WARNING_CONSOLE_ONLY,
    LOG_CONFIG_WARNING_FILE_ONLY,
    Log,
    PixelFormat,
    Stream,
    VmbSystem,
)

# ---------------------------------------------------------------------------
# Module-level Python logger (separate from the Vimba X internal log)
# ---------------------------------------------------------------------------
_log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    """All tuneable parameters for the camera session."""

    # Region of interest (pixels).  None → use camera default / full sensor.
    roi_offset_x: Optional[int] = None
    roi_offset_y: Optional[int] = None
    roi_width: Optional[int] = None
    roi_height: Optional[int] = None

    # Exposure time in microseconds.  None → leave at camera default.
    exposure_time_us: Optional[float] = None

    # Analogue gain (dB or camera units depending on model).  None → default.
    gain: Optional[float] = None

    # Brightness / black level (camera units).  None → default.
    brightness: Optional[float] = None

    # Target pixel format for acquisition.  None → keep camera default.
    pixel_format: Optional[PixelFormat] = PixelFormat.Mono8

    # Camera ID string (e.g. "DEV_1234…").  None → first detected camera.
    camera_id: Optional[str] = None

    # Software-trigger timeout for snapshot (seconds).
    snapshot_timeout_s: float = 5.0

    # Number of frame buffers pre-allocated for streaming.
    stream_buffer_count: int = 5

    # Optional path to save / load camera settings XML.
    settings_file: Optional[Path] = None

    # Vimba X log level constant.  None → no Vimba X-level logging.
    vmb_log_config: object = field(default=LOG_CONFIG_WARNING_CONSOLE_ONLY)


# ---------------------------------------------------------------------------
# Frame callback type alias
# ---------------------------------------------------------------------------
FrameCallback = Callable[[np.ndarray, float], None]
"""Receives (image_as_ndarray, timestamp_seconds)."""


# ---------------------------------------------------------------------------
# Hardware trigger enumerations  (thin wrappers around GenICam string values)
# ---------------------------------------------------------------------------

class TriggerActivation(str, Enum):
    """Edge / level on which the camera responds to the external signal.

    Supported values depend on camera model; check your camera's user manual.
    The string value is passed directly to ``TriggerActivation`` GenICam feature.
    """
    RISING_EDGE  = "RisingEdge"
    FALLING_EDGE = "FallingEdge"
    ANY_EDGE     = "AnyEdge"
    LEVEL_HIGH   = "LevelHigh"
    LEVEL_LOW    = "LevelLow"


class TriggerSelector(str, Enum):
    """Which acquisition event the trigger gates.

    ``FRAME_START`` is the most common choice and is supported by all Allied
    Vision cameras.  ``FRAME_BURST_START`` / ``ACQUISITION_START`` are
    available on select models.
    """
    FRAME_START         = "FrameStart"
    FRAME_BURST_START   = "FrameBurstStart"
    ACQUISITION_START   = "AcquisitionStart"
    EXPOSURE_ACTIVE     = "ExposureActive"


class AcquisitionMode(str, Enum):
    """Camera acquisition mode used while the hardware trigger is armed."""
    SINGLE_FRAME    = "SingleFrame"
    MULTI_FRAME     = "MultiFrame"
    CONTINUOUS      = "Continuous"


# ---------------------------------------------------------------------------
# Hardware trigger configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HardwareTriggerConfig:
    """All parameters that describe a hardware / external trigger session.

    Parameters
    ----------
    line:
        The camera GPIO input line name, e.g. ``"Line1"``, ``"Line2"``.
        Inspect your camera with Vimba X Viewer or call
        ``cam.list_hardware_trigger_lines()`` to see valid values.
    activation:
        Signal edge or level that fires the trigger.
    selector:
        The acquisition event being gated (almost always ``FRAME_START``).
    acquisition_mode:
        ``SINGLE_FRAME``  – camera waits for exactly one pulse then stops.
        ``CONTINUOUS``    – camera fires on every pulse until disarmed.
        ``MULTI_FRAME``   – camera fires on ``frame_count`` pulses.
    frame_count:
        Used only when ``acquisition_mode == MULTI_FRAME``.
    trigger_delay_us:
        Optional delay (µs) inserted between the trigger signal and the start
        of exposure.  ``None`` leaves the camera's current setting unchanged.
        Useful for compensating for known system latencies.
    debounce_us:
        Optional line-debounce time (µs).  ``None`` → camera default.
        Set this if the signal source is noisy (mechanical relay, long cable).
    timeout_s:
        Default wait timeout for ``wait_for_hardware_trigger()``.
    """
    line: str = "Line1"
    activation: TriggerActivation = TriggerActivation.RISING_EDGE
    selector: TriggerSelector = TriggerSelector.FRAME_START
    acquisition_mode: AcquisitionMode = AcquisitionMode.SINGLE_FRAME
    frame_count: int = 1
    trigger_delay_us: Optional[float] = None
    debounce_us: Optional[float] = None
    timeout_s: float = 10.0


# ---------------------------------------------------------------------------
# Main camera interface class
# ---------------------------------------------------------------------------
class AlliedVisionCamera:
    """
    Thread-safe interface to a single Allied Vision camera via VmbPy.

    Acquisition modes
    -----------------
    1. **Continuous** – free-run, non-critical timing, used for live display.
       ``start_continuous()`` / ``stop_continuous()``

    2. **Software-triggered snapshot** – single frame, timing-critical.
       Buffers pre-queued before trigger fires.
       ``take_snapshot()``

    3. **Hardware-triggered acquisition** – external electrical signal on a
       GPIO line gates each frame.  Supports single, multi, and continuous
       hardware-trigger modes.
       ``arm_hardware_trigger()`` / ``wait_for_hardware_trigger()`` /
       ``disarm_hardware_trigger()``

    Typical usage
    -------------
    >>> config = CameraConfig(exposure_time_us=10_000, gain=0.0)
    >>> with AlliedVisionCamera(config) as cam:
    ...     cam.start_continuous(callback=my_display_fn)
    ...     time.sleep(5)
    ...     cam.stop_continuous()
    ...     snapshot = cam.take_snapshot()
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self._config = config or CameraConfig()
        self._vmb: Optional[VmbSystem] = None
        self._cam: Optional[Camera] = None
        self._streaming = False
        self._stream_lock = threading.Lock()
        self._snapshot_event = threading.Event()
        self._snapshot_frame: Optional[np.ndarray] = None
        self._continuous_callback: Optional[FrameCallback] = None

        # Hardware trigger state
        self._hw_trigger_armed = False
        self._hw_trigger_config: Optional[HardwareTriggerConfig] = None
        self._hw_trigger_callback: Optional[FrameCallback] = None
        self._hw_frame_queue: list[np.ndarray] = []
        self._hw_frame_event = threading.Event()
        self._hw_frame_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context-manager / lifecycle
    # ------------------------------------------------------------------

    def open(self) -> "AlliedVisionCamera":
        """Initialise Vimba X, discover cameras, apply configuration."""
        self._vmb = VmbSystem.get_instance()
        self._vmb.__enter__()

        # Enable Vimba X internal logging
        if self._config.vmb_log_config is not None:
            self._vmb.enable_log(self._config.vmb_log_config)
            _log.info("Vimba X logging enabled.")

        # Register connection / disconnection hooks
        self._vmb.register_camera_change_handler(self._on_camera_change)

        # Select camera
        cameras = self._vmb.get_all_cameras()
        if not cameras:
            raise RuntimeError("No Allied Vision cameras detected.")

        if self._config.camera_id:
            matching = [c for c in cameras if c.get_id() == self._config.camera_id]
            if not matching:
                raise RuntimeError(
                    f"Camera '{self._config.camera_id}' not found. "
                    f"Available: {[c.get_id() for c in cameras]}"
                )
            self._cam = matching[0]
        else:
            self._cam = cameras[0]
            _log.info("No camera_id specified; using first detected: %s", self._cam.get_id())

        self._cam.__enter__()
        _log.info("Opened camera: %s  model: %s", self._cam.get_id(), self._cam.get_model())

        # Optionally load saved settings before applying overrides
        if self._config.settings_file and self._config.settings_file.exists():
            self.load_settings(self._config.settings_file)

        self._apply_config()
        return self

    def close(self) -> None:
        """Stop any ongoing acquisition and release all resources."""
        if self._hw_trigger_armed:
            self.disarm_hardware_trigger()

        if self._streaming:
            self.stop_continuous()

        if self._cam is not None:
            try:
                self._cam.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Exception while closing camera: %s", exc)
            self._cam = None

        if self._vmb is not None:
            try:
                self._vmb.disable_log()
                self._vmb.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Exception while shutting down VmbSystem: %s", exc)
            self._vmb = None

        _log.info("Camera interface closed.")

    def __enter__(self) -> "AlliedVisionCamera":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _apply_config(self) -> None:
        """Push CameraConfig values to the camera hardware."""
        cfg = self._config
        cam = self._cam

        # --- Pixel format ---
        if cfg.pixel_format is not None:
            supported = cam.get_pixel_formats()
            if cfg.pixel_format in supported:
                cam.set_pixel_format(cfg.pixel_format)
                _log.info("Pixel format set to %s", cfg.pixel_format)
            else:
                _log.warning(
                    "Requested pixel format %s not supported. Supported: %s",
                    cfg.pixel_format,
                    supported,
                )

        # --- Region of interest ---
        self._apply_roi(
            cfg.roi_offset_x,
            cfg.roi_offset_y,
            cfg.roi_width,
            cfg.roi_height,
        )

        # --- Exposure time ---
        if cfg.exposure_time_us is not None:
            self.set_exposure_time(cfg.exposure_time_us)

        # --- Gain ---
        if cfg.gain is not None:
            self.set_gain(cfg.gain)

        # --- Brightness / black level ---
        if cfg.brightness is not None:
            self.set_brightness(cfg.brightness)

    def _apply_roi(
        self,
        offset_x: Optional[int],
        offset_y: Optional[int],
        width: Optional[int],
        height: Optional[int],
    ) -> None:
        """
        Set the region of interest.  Parameters are applied in the safe order
        required by the GenICam standard: shrink before moving, expand after.
        """
        cam = self._cam
        try:
            # Always reset to full sensor first to avoid constraint violations
            cam.Width.set(cam.WidthMax.get())
            cam.Height.set(cam.HeightMax.get())
            cam.OffsetX.set(0)
            cam.OffsetY.set(0)

            if offset_x is not None:
                cam.OffsetX.set(offset_x)
            if offset_y is not None:
                cam.OffsetY.set(offset_y)
            if width is not None:
                cam.Width.set(width)
            if height is not None:
                cam.Height.set(height)

            _log.info(
                "ROI: OffsetX=%s OffsetY=%s Width=%s Height=%s",
                cam.OffsetX.get(),
                cam.OffsetY.get(),
                cam.Width.get(),
                cam.Height.get(),
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not set ROI: %s", exc)

    # ------------------------------------------------------------------
    # Public setters (can be called at runtime between frames)
    # ------------------------------------------------------------------

    def set_roi(
        self,
        offset_x: Optional[int] = None,
        offset_y: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Adjust the region of interest.  Pass None to leave a parameter unchanged."""
        if self._streaming:
            _log.warning("ROI changes while streaming may be ignored by some camera models.")
        self._apply_roi(offset_x, offset_y, width, height)
        # Persist in config so reopen() restores the same ROI
        if offset_x is not None:
            self._config.roi_offset_x = offset_x
        if offset_y is not None:
            self._config.roi_offset_y = offset_y
        if width is not None:
            self._config.roi_width = width
        if height is not None:
            self._config.roi_height = height

    def set_exposure_time(self, microseconds: float) -> None:
        """Set exposure time in microseconds."""
        try:
            feat = self._cam.ExposureTime
            min_val = feat.get_range()[0]
            max_val = feat.get_range()[1]
            clamped = max(min_val, min(max_val, microseconds))
            if clamped != microseconds:
                _log.warning(
                    "Exposure %g µs out of range [%g, %g]; clamped to %g.",
                    microseconds, min_val, max_val, clamped,
                )
            feat.set(clamped)
            _log.info("Exposure time set to %.1f µs", clamped)
            self._config.exposure_time_us = clamped
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not set exposure time: %s", exc)

    def set_gain(self, value: float) -> None:
        """Set analogue gain (dB or camera-specific units)."""
        try:
            feat = self._cam.Gain
            min_val, max_val = feat.get_range()
            clamped = max(min_val, min(max_val, value))
            if clamped != value:
                _log.warning(
                    "Gain %g out of range [%g, %g]; clamped to %g.",
                    value, min_val, max_val, clamped,
                )
            feat.set(clamped)
            _log.info("Gain set to %.3f", clamped)
            self._config.gain = clamped
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not set gain: %s", exc)

    def set_brightness(self, value: float) -> None:
        """
        Set brightness / black level.  The exact feature name varies by model;
        we try 'Brightness' first, then 'BlackLevel'.
        """
        cam = self._cam
        for feat_name in ("Brightness", "BlackLevel"):
            try:
                feat = getattr(cam, feat_name)
                feat.set(value)
                _log.info("%s set to %.3f", feat_name, value)
                self._config.brightness = value
                return
            except Exception:  # noqa: BLE001
                continue
        _log.warning("Could not set brightness / black level (feature not found or not writable).")

    def set_pixel_format(self, fmt: PixelFormat) -> None:
        """Change the pixel format (must not be called while streaming)."""
        if self._streaming:
            raise RuntimeError("Cannot change pixel format while streaming.")
        supported = self._cam.get_pixel_formats()
        if fmt not in supported:
            raise ValueError(f"Pixel format {fmt} not supported. Supported: {supported}")
        self._cam.set_pixel_format(fmt)
        self._config.pixel_format = fmt
        _log.info("Pixel format changed to %s", fmt)

    # ------------------------------------------------------------------
    # Continuous streaming  (non-critical timing – for live display)
    # ------------------------------------------------------------------

    def start_continuous(self, callback: Optional[FrameCallback] = None) -> None:
        """
        Start asynchronous frame acquisition.

        Parameters
        ----------
        callback:
            Optional callable ``fn(image: np.ndarray, timestamp: float) -> None``
            invoked on every incoming frame.  Keep it short; heavy processing
            should be handed off to another thread.
        """
        with self._stream_lock:
            if self._streaming:
                _log.warning("start_continuous() called while already streaming.")
                return

            self._continuous_callback = callback
            cam = self._cam

            # Put camera in free-run (no external trigger)
            try:
                cam.TriggerMode.set("Off")
                cam.AcquisitionMode.set("Continuous")
            except Exception as exc:  # noqa: BLE001
                _log.warning("Could not set continuous acquisition mode: %s", exc)

            cam.start_streaming(self._continuous_frame_handler, buffer_count=self._config.stream_buffer_count)
            self._streaming = True
            _log.info("Continuous streaming started.")

    def stop_continuous(self) -> None:
        """Stop asynchronous frame acquisition."""
        with self._stream_lock:
            if not self._streaming:
                return
            try:
                self._cam.stop_streaming()
            except Exception as exc:  # noqa: BLE001
                _log.warning("Exception stopping stream: %s", exc)
            self._streaming = False
            self._continuous_callback = None
            _log.info("Continuous streaming stopped.")

    def _continuous_frame_handler(self, cam: Camera, stream: Stream, frame: Frame) -> None:
        """Internal callback invoked by VmbPy for each incoming frame."""
        if frame.get_status() != FrameStatus.Complete:
            _log.warning("Incomplete frame received (status=%s). Skipping.", frame.get_status())
            cam.queue_frame(frame)
            return

        try:
            img = self._frame_to_ndarray(frame)
            ts = frame.get_timestamp() / 1e9  # nanoseconds → seconds
            if self._continuous_callback is not None:
                self._continuous_callback(img, ts)
        except Exception as exc:  # noqa: BLE001
            _log.error("Error in continuous frame handler: %s", exc)
        finally:
            cam.queue_frame(frame)

    # ------------------------------------------------------------------
    # Software-triggered snapshot  (timing-critical)
    # ------------------------------------------------------------------

    def take_snapshot(self) -> np.ndarray:
        """
        Arm the camera for a single software-triggered acquisition, fire the
        trigger immediately, wait for the frame, and return it as an ndarray.

        This path is optimised for minimal latency between the trigger command
        and the start of exposure.  The camera is placed in SingleFrame /
        Software-trigger mode, the streaming pipeline is pre-started (so
        buffers are already queued in the driver), and only then is the trigger
        command issued.

        Returns
        -------
        np.ndarray
            The captured image.

        Raises
        ------
        RuntimeError
            If continuous streaming is currently active (stop it first), or if
            the frame is not received within ``config.snapshot_timeout_s``.
        """
        if self._streaming:
            raise RuntimeError(
                "stop_continuous() before calling take_snapshot()."
            )

        cam = self._cam
        self._snapshot_frame = None
        self._snapshot_event.clear()

        # Configure software trigger
        cam.TriggerSource.set("Software")
        cam.TriggerSelector.set("FrameStart")
        cam.TriggerMode.set("On")
        cam.AcquisitionMode.set("SingleFrame")

        # Pre-start streaming so the frame buffer is already waiting in the driver
        cam.start_streaming(
            self._snapshot_frame_handler,
            buffer_count=1,
        )

        try:
            # -------------------------------------------------------
            # FIRE TRIGGER  ← minimise code between here and .run()
            # -------------------------------------------------------
            cam.TriggerSoftware.run()
            # -------------------------------------------------------

            acquired = self._snapshot_event.wait(timeout=self._config.snapshot_timeout_s)
        finally:
            cam.stop_streaming()
            # Restore to free-run so the camera is ready for next use
            try:
                cam.TriggerMode.set("Off")
                cam.AcquisitionMode.set("Continuous")
            except Exception as exc:  # noqa: BLE001
                _log.warning("Could not restore acquisition mode after snapshot: %s", exc)

        if not acquired or self._snapshot_frame is None:
            raise RuntimeError(
                f"Snapshot timed out after {self._config.snapshot_timeout_s} s."
            )

        _log.info("Snapshot acquired successfully.")
        return self._snapshot_frame

    def _snapshot_frame_handler(self, cam: Camera, stream: Stream, frame: Frame) -> None:
        """Internal callback for software-triggered single frame."""
        if frame.get_status() != FrameStatus.Complete:
            _log.warning(
                "Snapshot frame incomplete (status=%s).", frame.get_status()
            )
            cam.queue_frame(frame)
            return

        try:
            self._snapshot_frame = self._frame_to_ndarray(frame)
        except Exception as exc:  # noqa: BLE001
            _log.error("Error converting snapshot frame: %s", exc)
        finally:
            self._snapshot_event.set()
            # Do NOT re-queue; we only want one frame.

    # ------------------------------------------------------------------
    # Hardware / external trigger acquisition
    # ------------------------------------------------------------------

    def list_hardware_trigger_lines(self) -> list[str]:
        """
        Return the GPIO line names that this camera exposes as trigger inputs.

        Use the returned strings as ``HardwareTriggerConfig.line``.
        Example output: ``['Line0', 'Line1', 'Line2']``
        """
        cam = self._cam
        try:
            # LineSelector is an enumeration feature; its entries are the line names.
            return list(cam.LineSelector.get_all_entries())
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not enumerate GPIO lines: %s", exc)
            return []

    def arm_hardware_trigger(
        self,
        hw_config: HardwareTriggerConfig,
        callback: Optional[FrameCallback] = None,
    ) -> None:
        """
        Configure the camera for hardware / external triggering and start the
        acquisition pipeline so it is ready the instant the signal arrives.

        The camera will expose a frame on every pulse (or level, depending on
        ``hw_config.activation``) detected on ``hw_config.line``.

        Parameters
        ----------
        hw_config:
            Hardware trigger parameters (line, edge, selector, mode, …).
        callback:
            Optional ``fn(image: np.ndarray, timestamp: float) -> None``
            called from the VmbPy callback thread each time a triggered frame
            arrives.  For ``SINGLE_FRAME`` mode you may prefer the blocking
            ``wait_for_hardware_trigger()`` instead.

        Raises
        ------
        RuntimeError
            If continuous streaming or another hardware trigger session is
            already active.

        Notes on latency
        ----------------
        The acquisition pipeline (frame buffers) is started *before* this
        method returns.  This means the camera is already waiting in hardware
        when ``arm_hardware_trigger`` completes, so the interval between the
        physical pulse and the start of exposure contains only:
          - camera input-circuit propagation delay  (~µs, fixed, camera-spec)
          - optional ``trigger_delay_us`` you configure here
        No Python / OS scheduling jitter is in the critical path.
        """
        if self._streaming:
            raise RuntimeError(
                "stop_continuous() before calling arm_hardware_trigger()."
            )
        if self._hw_trigger_armed:
            raise RuntimeError(
                "disarm_hardware_trigger() before re-arming."
            )

        cam = self._cam
        self._hw_trigger_config = hw_config
        self._hw_trigger_callback = callback

        with self._hw_frame_lock:
            self._hw_frame_queue.clear()
        self._hw_frame_event.clear()

        # --- Configure GPIO line direction ---
        try:
            cam.LineSelector.set(hw_config.line)
            cam.LineMode.set("Input")
            _log.info("GPIO %s set to Input.", hw_config.line)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "Could not configure line direction for %s: %s  "
                "(some cameras configure lines automatically).",
                hw_config.line, exc,
            )

        # --- Optional line debounce ---
        if hw_config.debounce_us is not None:
            try:
                cam.LineDebouncerTime.set(hw_config.debounce_us)
                _log.info("Line debounce set to %.1f µs.", hw_config.debounce_us)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Could not set line debounce: %s", exc)

        # --- Trigger selector and source ---
        try:
            cam.TriggerSelector.set(hw_config.selector.value)
            cam.TriggerSource.set(hw_config.line)
            cam.TriggerActivation.set(hw_config.activation.value)
            cam.TriggerMode.set("On")
            _log.info(
                "Hardware trigger: selector=%s  source=%s  activation=%s",
                hw_config.selector.value,
                hw_config.line,
                hw_config.activation.value,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to configure trigger features: {exc}") from exc

        # --- Optional trigger delay ---
        if hw_config.trigger_delay_us is not None:
            try:
                cam.TriggerDelay.set(hw_config.trigger_delay_us)
                _log.info("Trigger delay set to %.1f µs.", hw_config.trigger_delay_us)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Could not set trigger delay: %s", exc)

        # --- Acquisition mode ---
        try:
            cam.AcquisitionMode.set(hw_config.acquisition_mode.value)
            if hw_config.acquisition_mode == AcquisitionMode.MULTI_FRAME:
                cam.AcquisitionFrameCount.set(hw_config.frame_count)
                _log.info("Multi-frame count: %d", hw_config.frame_count)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to set acquisition mode: {exc}") from exc

        # --- Start streaming pipeline (buffers queued in driver NOW) ---
        buffer_count = max(
            self._config.stream_buffer_count,
            hw_config.frame_count if hw_config.acquisition_mode == AcquisitionMode.MULTI_FRAME else 1,
        )
        cam.start_streaming(self._hw_frame_handler, buffer_count=buffer_count)
        self._hw_trigger_armed = True

        _log.info(
            "Hardware trigger armed on %s (%s).  "
            "Camera is waiting for external signal.",
            hw_config.line,
            hw_config.activation.value,
        )

    def wait_for_hardware_trigger(
        self,
        timeout_s: Optional[float] = None,
    ) -> np.ndarray:
        """
        Block until the hardware trigger fires and a complete frame is received,
        then return the image as a NumPy array.

        Must be called after ``arm_hardware_trigger()``.

        Parameters
        ----------
        timeout_s:
            Seconds to wait.  ``None`` uses ``hw_config.timeout_s``.

        Returns
        -------
        np.ndarray
            The triggered frame.

        Raises
        ------
        RuntimeError
            If the camera was not armed or the wait times out.
        """
        if not self._hw_trigger_armed:
            raise RuntimeError("Call arm_hardware_trigger() first.")

        t = timeout_s if timeout_s is not None else self._hw_trigger_config.timeout_s
        _log.info("Waiting for hardware trigger (timeout=%.1f s) …", t)

        got_frame = self._hw_frame_event.wait(timeout=t)
        if not got_frame:
            raise RuntimeError(
                f"Hardware trigger timed out after {t} s.  "
                "Check signal source and GPIO wiring."
            )

        with self._hw_frame_lock:
            if not self._hw_frame_queue:
                raise RuntimeError("Trigger event set but frame queue is empty.")
            img = self._hw_frame_queue.pop(0)
            if not self._hw_frame_queue:
                self._hw_frame_event.clear()   # reset for next wait in CONTINUOUS mode

        _log.info("Hardware-triggered frame received.")
        return img

    def disarm_hardware_trigger(self) -> None:
        """
        Stop the hardware trigger acquisition pipeline and restore the camera
        to free-run (trigger off) so it is ready for the next use.
        """
        if not self._hw_trigger_armed:
            _log.debug("disarm_hardware_trigger() called while not armed; ignored.")
            return

        try:
            self._cam.stop_streaming()
        except Exception as exc:  # noqa: BLE001
            _log.warning("Exception while stopping hardware trigger stream: %s", exc)

        try:
            self._cam.TriggerMode.set("Off")
            self._cam.AcquisitionMode.set("Continuous")
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not restore acquisition mode: %s", exc)

        self._hw_trigger_armed = False
        self._hw_trigger_config = None
        self._hw_trigger_callback = None
        with self._hw_frame_lock:
            self._hw_frame_queue.clear()
        self._hw_frame_event.clear()

        _log.info("Hardware trigger disarmed.")

    def _hw_frame_handler(self, cam: Camera, stream: Stream, frame: Frame) -> None:
        """Internal VmbPy callback for hardware-triggered frames."""
        if frame.get_status() != FrameStatus.Complete:
            _log.warning(
                "Hardware-triggered frame incomplete (status=%s). Discarding.",
                frame.get_status(),
            )
            cam.queue_frame(frame)
            return

        try:
            img = self._frame_to_ndarray(frame)
            ts = frame.get_timestamp() / 1e9

            # Push into queue for wait_for_hardware_trigger()
            with self._hw_frame_lock:
                self._hw_frame_queue.append(img)
            self._hw_frame_event.set()

            # Also fire the user callback if provided
            if self._hw_trigger_callback is not None:
                try:
                    self._hw_trigger_callback(img, ts)
                except Exception as exc:  # noqa: BLE001
                    _log.error("Error in hardware trigger callback: %s", exc)

        except Exception as exc:  # noqa: BLE001
            _log.error("Error processing hardware-triggered frame: %s", exc)
        finally:
            # Re-queue unless SingleFrame mode (camera stops itself after one frame)
            cfg = self._hw_trigger_config
            if cfg is not None and cfg.acquisition_mode != AcquisitionMode.SINGLE_FRAME:
                cam.queue_frame(frame)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _frame_to_ndarray(self, frame: Frame) -> np.ndarray:
        """Convert a VmbPy Frame to a NumPy ndarray (BGR for colour, 2-D for mono)."""
        fmt = frame.get_pixel_format()
        # Convert to Mono8 or BGR8 so OpenCV can handle it
        if fmt in (PixelFormat.Mono8,):
            return frame.as_numpy_ndarray()
        elif fmt in (PixelFormat.Bgr8,):
            return frame.as_opencv_image()
        else:
            # Generic conversion to Mono8 for unknown formats
            frame.convert_pixel_format(PixelFormat.Mono8)
            return frame.as_numpy_ndarray()

    def get_camera_info(self) -> dict:
        """Return a dict of static camera properties."""
        cam = self._cam
        return {
            "id": cam.get_id(),
            "model": cam.get_model(),
            "serial": cam.get_serial(),
            "interface_id": cam.get_interface_id(),
            "pixel_formats": [str(f) for f in cam.get_pixel_formats()],
            "current_pixel_format": str(cam.get_pixel_format()),
            "width": cam.Width.get(),
            "height": cam.Height.get(),
            "width_max": cam.WidthMax.get(),
            "height_max": cam.HeightMax.get(),
        }

    def read_temperature(self) -> Optional[float]:
        """Return the camera's device temperature in °C, or None if unsupported."""
        try:
            return self._cam.DeviceTemperature.get()
        except Exception:  # noqa: BLE001
            _log.debug("DeviceTemperature not available on this camera.")
            return None

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def save_settings(self, path: Path) -> None:
        """Save all camera feature values to an XML file on the host PC."""
        path = Path(path)
        self._cam.save_settings(str(path), PersistType.All)
        _log.info("Camera settings saved to %s", path)

    def load_settings(self, path: Path) -> None:
        """Load camera feature values from a previously saved XML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Settings file not found: {path}")
        self._cam.load_settings(str(path), PersistType.All)
        _log.info("Camera settings loaded from %s", path)

    # ------------------------------------------------------------------
    # Camera connection / disconnection callback
    # ------------------------------------------------------------------

    @staticmethod
    def _on_camera_change(camera: Camera, state) -> None:
        _log.warning("Camera change detected — device=%s, state=%s", camera.get_id(), state)


# ---------------------------------------------------------------------------
# Convenience: simple OpenCV live-view loop
# ---------------------------------------------------------------------------

def live_view(cam: AlliedVisionCamera, window_title: str = "Live View") -> None:
    """
    Blocking live-view using OpenCV.  Press 'q' or Escape to quit.
    Runs the continuous stream in the background and renders frames
    via cv2.imshow on the calling thread.
    """
    latest: list[Optional[np.ndarray]] = [None]
    lock = threading.Lock()

    def _cb(img: np.ndarray, _ts: float) -> None:
        with lock:
            latest[0] = img.copy()

    cam.start_continuous(callback=_cb)
    _log.info("Live view started. Press 'q' or <Esc> to quit.")

    try:
        while True:
            with lock:
                frame = latest[0]
            if frame is not None:
                cv2.imshow(window_title, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):  # q or Escape
                break
    finally:
        cam.stop_continuous()
        cv2.destroyWindow(window_title)
        _log.info("Live view closed.")


# ---------------------------------------------------------------------------
# Demo / smoke-test  (run directly: python camera_interface.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Build a configuration – adjust values for your camera
    config = CameraConfig(
        exposure_time_us=20_000,   # 20 ms
        gain=0.0,
        brightness=None,           # leave at camera default
        roi_offset_x=0,
        roi_offset_y=0,
        roi_width=None,            # None → full sensor width
        roi_height=None,
        pixel_format=PixelFormat.Mono8,
        snapshot_timeout_s=5.0,
    )

    with AlliedVisionCamera(config) as cam:
        print("Camera info:", cam.get_camera_info())
        temp = cam.read_temperature()
        if temp is not None:
            print(f"Sensor temperature: {temp:.1f} °C")

        # Inspect available GPIO lines before choosing one
        lines = cam.list_hardware_trigger_lines()
        print("Available GPIO lines:", lines)

        # ---- Live view (non-blocking in script; blocks here for demo) ----
        live_view(cam, window_title="Allied Vision – Live View")

        # ---- Software-triggered snapshot ----
        print("Taking software-triggered snapshot …")
        snapshot = cam.take_snapshot()
        cv2.imwrite("snapshot_software.png", snapshot)
        print(f"Software snapshot saved: {snapshot.shape}")

        # ---- Hardware-triggered single-frame acquisition ----
        # Adjust 'line' and 'activation' to match your wiring.
        hw_cfg = HardwareTriggerConfig(
            line="Line1",
            activation=TriggerActivation.RISING_EDGE,
            selector=TriggerSelector.FRAME_START,
            acquisition_mode=AcquisitionMode.SINGLE_FRAME,
            trigger_delay_us=0.0,     # no intentional delay
            debounce_us=10.0,         # 10 µs debounce for clean signals
            timeout_s=10.0,
        )

        print("Arming hardware trigger on Line1 (rising edge) …")
        cam.arm_hardware_trigger(hw_cfg)
        print("Camera is armed.  Send a rising edge on Line1 within 10 s.")

        try:
            hw_frame = cam.wait_for_hardware_trigger(timeout_s=10.0)
            cv2.imwrite("snapshot_hardware.png", hw_frame)
            print(f"Hardware-triggered frame saved: {hw_frame.shape}")
        except RuntimeError as e:
            print(f"Hardware trigger error: {e}")
        finally:
            cam.disarm_hardware_trigger()

        # ---- Hardware-triggered continuous acquisition example ----
        # Fires a callback on every external pulse until disarmed.
        hw_cfg_cont = HardwareTriggerConfig(
            line="Line1",
            activation=TriggerActivation.RISING_EDGE,
            acquisition_mode=AcquisitionMode.CONTINUOUS,
            timeout_s=30.0,
        )

        received: list[int] = [0]

        def _hw_cb(img: np.ndarray, ts: float) -> None:
            received[0] += 1
            print(f"  HW frame #{received[0]}  ts={ts:.6f}  shape={img.shape}")

        print("\nArming hardware trigger in CONTINUOUS mode for 10 s …")
        cam.arm_hardware_trigger(hw_cfg_cont, callback=_hw_cb)
        time.sleep(10)
        cam.disarm_hardware_trigger()
        print(f"Total hardware-triggered frames received: {received[0]}")

        # ---- Runtime parameter adjustment ----
        cam.set_exposure_time(5_000)   # 5 ms
        cam.set_gain(6.0)
        cam.set_roi(offset_x=100, offset_y=100, width=640, height=480)

    print("Done.")
    sys.exit(0)