"""
waveforms_ads.py
================
Python class for controlling Digilent Analog Discovery / WaveForms hardware
via the DWF (dwf.dll / libdwf.so) C library.

Wraps the most important instrument groups from the WaveForms SDK:
  - Device enumeration & control
  - Analog In  (Oscilloscope)
  - Analog Out (Arbitrary Waveform Generator)
  - Analog I/O (Power supplies / system monitor)
  - Digital I/O

Requirements
------------
  pip install numpy
  Digilent WaveForms + Adept Runtime installed (provides dwf.dll / libdwf.so)

Quick start
-----------
    from waveforms_ads import WaveFormsADS

    with WaveFormsADS() as dev:
        # Single voltage sample on Ch0
        v = dev.analog_in_read_sample(channel=0)
        print(f"Ch0 = {v:.4f} V")

        # Generate a 1 kHz sine wave at 1 V amplitude on Ch0
        dev.analog_out_set_sine(channel=0, freq_hz=1_000, amplitude_v=1.0)
        dev.analog_out_start(channel=0)

        # Triggered oscilloscope capture
        data = dev.analog_in_capture(
            channel=0,
            sample_rate_hz=1e6,
            buffer_size=4096,
            trigger_level_v=0.1,
        )
"""

import ctypes
import os
import sys
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# DWF constant definitions (from dwf.h)
# ---------------------------------------------------------------------------

# Device states
DwfStateReady    = 0
DwfStateConfig   = 4
DwfStatePrefill  = 5
DwfStateArmed    = 1
DwfStateWait     = 7
DwfStateTriggered = 3
DwfStateRunning  = 3
DwfStateDone     = 2

# Acquisition modes
acqmodeSingle     = 0
acqmodeScanShift  = 1
acqmodeScanScreen = 2
acqmodeRecord     = 3
acqmodeSingle1    = 5

# Waveform functions
funcDC          = 0
funcSine        = 1
funcSquare      = 2
funcTriangle    = 3
funcRampUp      = 4
funcRampDown    = 5
funcNoise       = 6
funcPulse       = 7
funcTrapezium   = 8
funcSinePower   = 9
funcCustom      = 30
funcPlay        = 31

# Analog Out nodes
AnalogOutNodeCarrier = 0
AnalogOutNodeFM      = 1
AnalogOutNodeAM      = 2

# Trigger sources
trigsrcNone              = 0
trigsrcPC                = 1
trigsrcDetectorAnalogIn  = 2
trigsrcDetectorDigitalIn = 3
trigsrcAnalogIn          = 4
trigsrcDigitalIn         = 5
trigsrcExternal1         = 11

# Trigger types
trigtypeEdge       = 0
trigtypePulse      = 1
trigtypeTransition = 2
trigtypeWindow     = 3

# Trigger slopes
DwfTriggerSlopeRise   = 0
DwfTriggerSlopeFall   = 1
DwfTriggerSlopeEither = 2

# Channel filters
filterDecimate   = 0
filterAverage    = 1
filterMinMax     = 2

# Coupling
DwfAnalogCouplingDC = 0
DwfAnalogCouplingAC = 1

# Invalid handle sentinel
hdwfNone = ctypes.c_int(0)

# AutoConfigure modes
AUTOCFG_DISABLE  = 0
AUTOCFG_ENABLE   = 1
AUTOCFG_DYNAMIC  = 3


# ---------------------------------------------------------------------------
# Library loader
# ---------------------------------------------------------------------------

def _load_dwf() -> ctypes.CDLL:
    """Load the DWF shared library for the current platform."""
    if sys.platform == "win32":
        # Try both 64-bit and 32-bit paths
        candidates = [
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "dwf.dll"),
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "SysWOW64", "dwf.dll"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return ctypes.cdll.LoadLibrary(path)
        raise OSError("dwf.dll not found. Install Digilent WaveForms.")
    elif sys.platform == "darwin":
        return ctypes.cdll.LoadLibrary(
            "/Library/Frameworks/dwf.framework/dwf"
        )
    else:  # Linux
        return ctypes.cdll.LoadLibrary("libdwf.so")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DWFError(Exception):
    """Raised when a DWF API call fails."""

class TriggerWithoutChannelError(Exception):
    """Raised when a trigger has been set with no channel to trigger on."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WaveFormsADS:
    """
    High-level Python controller for Digilent WaveForms Analog Discovery
    Series (ADS) hardware.

    Can be used as a context manager (``with WaveFormsADS() as dev: ...``).

    Parameters
    ----------
    device_index : int
        0-based index of the device to open.  Pass -1 (default) to
        automatically open the first detected device.
    config_index : int
        Device configuration index.  -1 means use the default.
    auto_configure : int
        0 = disable AutoConfigure (fastest, manual configure calls needed),
        1 = enable (default behaviour), 3 = dynamic.
    """

    def __init__(
        self,
        device_index: int = -1,
        config_index: int = -1,
        auto_configure: int = AUTOCFG_ENABLE,
    ) -> None:
        self._dwf = _load_dwf()
        self._hdwf = ctypes.c_int(0)
        self._auto_configure = auto_configure
        self._open(device_index, config_index)

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    def __enter__(self) -> "WaveFormsADS":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check(self, ret: int, context: str = "") -> None:
        """Raise DWFError if a DWF API call returned 0 (failure)."""
        if ret == 0:
            msg_buf = ctypes.create_string_buffer(512)
            self._dwf.FDwfGetLastErrorMsg(msg_buf)
            err = msg_buf.value.decode(errors="replace").strip()
            raise DWFError(f"DWF call failed [{context}]: {err}")

    def _state_name(self, state: int) -> str:
        names = {
            DwfStateReady: "Ready", DwfStateConfig: "Config",
            DwfStatePrefill: "Prefill", DwfStateArmed: "Armed",
            DwfStateDone: "Done",
        }
        return names.get(state, f"State({state})")

    # ------------------------------------------------------------------
    # Device enumeration (static helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def enumerate() -> List[dict]:
        """
        Return a list of dicts describing every connected, compatible device.

        Each dict has keys: ``index``, ``name``, ``user_name``, ``serial``,
        ``is_open``.
        """
        dwf = _load_dwf()
        n_dev = ctypes.c_int(0)
        dwf.FDwfEnum(0, ctypes.byref(n_dev))  # enumfilterAll = 0

        devices = []
        for i in range(n_dev.value):
            dev_name = ctypes.create_string_buffer(32)
            user_name = ctypes.create_string_buffer(32)
            serial = ctypes.create_string_buffer(32)
            is_open = ctypes.c_int(0)

            dwf.FDwfEnumDeviceName(i, dev_name)
            dwf.FDwfEnumUserName(i, user_name)
            dwf.FDwfEnumSN(i, serial)
            dwf.FDwfEnumDeviceIsOpened(i, ctypes.byref(is_open))

            devices.append({
                "index":     i,
                "name":      dev_name.value.decode(errors="replace"),
                "user_name": user_name.value.decode(errors="replace"),
                "serial":    serial.value.decode(errors="replace"),
                "is_open":   bool(is_open.value),
            })
        return devices

    # ------------------------------------------------------------------
    # Device open / close
    # ------------------------------------------------------------------

    def _open(self, device_index: int, config_index: int) -> None:
        hdwf = ctypes.c_int(0)
        if config_index >= 0:
            ret = self._dwf.FDwfDeviceConfigOpen(
                device_index, config_index, ctypes.byref(hdwf)
            )
        else:
            ret = self._dwf.FDwfDeviceOpen(device_index, ctypes.byref(hdwf))
        self._check(ret, "FDwfDeviceOpen")
        if hdwf.value == 0:
            raise DWFError("No device found or device could not be opened.")
        self._hdwf = hdwf

        # Apply AutoConfigure preference
        self._dwf.FDwfDeviceAutoConfigureSet(self._hdwf, self._auto_configure)

    def close(self) -> None:
        """Close the device handle."""
        if self._hdwf.value != 0:
            self._dwf.FDwfDeviceClose(self._hdwf)
            self._hdwf = ctypes.c_int(0)

    def reset(self) -> None:
        """Reset all instrument parameters to defaults."""
        self._check(self._dwf.FDwfDeviceReset(self._hdwf), "FDwfDeviceReset")

    # ------------------------------------------------------------------
    # System / DWF version
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        """Return the DWF library version string (e.g. '3.21.2')."""
        buf = ctypes.create_string_buffer(32)
        self._dwf.FDwfGetVersion(buf)
        return buf.value.decode()

    def get_last_error(self) -> Tuple[int, str]:
        """Return (error_code, error_message) from the last failed call."""
        erc = ctypes.c_int(0)
        msg = ctypes.create_string_buffer(512)
        self._dwf.FDwfGetLastError(ctypes.byref(erc))
        self._dwf.FDwfGetLastErrorMsg(msg)
        return erc.value, msg.value.decode(errors="replace").strip()

    # ------------------------------------------------------------------
    # Analog In – Oscilloscope
    # ------------------------------------------------------------------

    def analog_in_channel_count(self) -> int:
        """Return the number of Analog In channels."""
        n = ctypes.c_int(0)
        self._check(
            self._dwf.FDwfAnalogInChannelCount(self._hdwf, ctypes.byref(n)),
            "FDwfAnalogInChannelCount",
        )
        return n.value

    def analog_in_reset(self) -> None:
        """Reset Analog In instrument to defaults."""
        self._check(self._dwf.FDwfAnalogInReset(self._hdwf), "FDwfAnalogInReset")

    def analog_in_configure(self, reconfigure: bool = True, start: bool = True) -> None:
        """Configure (and optionally start) the Analog In acquisition."""
        self._check(
            self._dwf.FDwfAnalogInConfigure(self._hdwf, int(reconfigure), int(start)),
            "FDwfAnalogInConfigure",
        )

    def analog_in_status(self, read_data: bool = True) -> int:
        """Poll Analog In status.  Returns the DwfState integer."""
        sts = ctypes.c_byte(0)
        self._check(
            self._dwf.FDwfAnalogInStatus(self._hdwf, int(read_data), ctypes.byref(sts)),
            "FDwfAnalogInStatus",
        )
        return int(sts.value)

    # --- Configuration --------------------------------------------------

    def analog_in_set_sample_rate(self, hz: float) -> None:
        """Set the ADC sample rate in Hz."""
        self._check(
            self._dwf.FDwfAnalogInFrequencySet(self._hdwf, ctypes.c_double(hz)),
            "FDwfAnalogInFrequencySet",
        )

    def analog_in_get_sample_rate(self) -> float:
        """Return the configured ADC sample rate in Hz."""
        hz = ctypes.c_double(0.0)
        self._check(
            self._dwf.FDwfAnalogInFrequencyGet(self._hdwf, ctypes.byref(hz)),
            "FDwfAnalogInFrequencyGet",
        )
        return hz.value

    def analog_in_set_buffer_size(self, size: int) -> None:
        """Set the acquisition buffer size (number of samples)."""
        self._check(
            self._dwf.FDwfAnalogInBufferSizeSet(self._hdwf, size),
            "FDwfAnalogInBufferSizeSet",
        )

    def analog_in_get_buffer_size(self) -> int:
        """Return the current acquisition buffer size."""
        n = ctypes.c_int(0)
        self._check(
            self._dwf.FDwfAnalogInBufferSizeGet(self._hdwf, ctypes.byref(n)),
            "FDwfAnalogInBufferSizeGet",
        )
        return n.value

    def analog_in_set_acquisition_mode(self, mode: int = acqmodeSingle) -> None:
        """Set acquisition mode (acqmodeSingle, acqmodeRecord, etc.)."""
        self._check(
            self._dwf.FDwfAnalogInAcquisitionModeSet(self._hdwf, mode),
            "FDwfAnalogInAcquisitionModeSet",
        )

    # --- Channel settings -----------------------------------------------

    def analog_in_channel_enable(self, channel: int, enable: bool = True) -> None:
        """Enable or disable an Analog In channel (0-based index)."""
        self._check(
            self._dwf.FDwfAnalogInChannelEnableSet(self._hdwf, channel, int(enable)),
            "FDwfAnalogInChannelEnableSet",
        )

    def analog_in_set_range(self, channel: int, voltage_range: float) -> None:
        """Set the peak-to-peak voltage range for a channel (e.g. 5.0 for ±2.5 V)."""
        self._check(
            self._dwf.FDwfAnalogInChannelRangeSet(
                self._hdwf, channel, ctypes.c_double(voltage_range)
            ),
            "FDwfAnalogInChannelRangeSet",
        )

    def analog_in_get_range(self, channel: int) -> float:
        """Return the actual voltage range for the channel."""
        v = ctypes.c_double(0.0)
        self._check(
            self._dwf.FDwfAnalogInChannelRangeGet(self._hdwf, channel, ctypes.byref(v)),
            "FDwfAnalogInChannelRangeGet",
        )
        return v.value

    def analog_in_set_offset(self, channel: int, offset_v: float) -> None:
        """Set the channel DC offset in volts."""
        self._check(
            self._dwf.FDwfAnalogInChannelOffsetSet(
                self._hdwf, channel, ctypes.c_double(offset_v)
            ),
            "FDwfAnalogInChannelOffsetSet",
        )

    def analog_in_set_attenuation(self, channel: int, attenuation: float) -> None:
        """Set the channel attenuation as a unitless scaling factor."""
        self._check(
            self._dwf.FDwfAnalogInChannelAttenuationSet(
                self._hdwf, channel, ctypes.c_double(attenuation)
            ),
            "FDwfAnalogInChannelAttenuationSet",
        )

    def analog_in_set_coupling(self, channel: int, coupling: int = DwfAnalogCouplingDC) -> None:
        """Set channel coupling: DwfAnalogCouplingDC (0) or DwfAnalogCouplingAC (1)."""
        self._check(
            self._dwf.FDwfAnalogInChannelCouplingSet(self._hdwf, channel, coupling),
            "FDwfAnalogInChannelCouplingSet",
        )

    def analog_in_set_filter(self, channel: int, filt: int = filterDecimate) -> None:
        """Set the acquisition filter for a channel (filterDecimate / filterAverage / filterMinMax)."""
        self._check(
            self._dwf.FDwfAnalogInChannelFilterSet(self._hdwf, channel, filt),
            "FDwfAnalogInChannelFilterSet",
        )

    # --- Trigger --------------------------------------------------------

    def analog_in_set_trigger_source(self, source: int = trigsrcDetectorAnalogIn) -> None:
        """Set the Analog In trigger source."""
        self._check(
            self._dwf.FDwfAnalogInTriggerSourceSet(self._hdwf, source),
            "FDwfAnalogInTriggerSourceSet",
        )

    def analog_in_set_trigger_channel(self, channel: int) -> None:
        """Set the channel used by the trigger detector."""
        self._check(
            self._dwf.FDwfAnalogInTriggerChannelSet(self._hdwf, channel),
            "FDwfAnalogInTriggerChannelSet",
        )

    def analog_in_set_trigger_type(self, trig_type: int = trigtypeEdge) -> None:
        """Set trigger type (trigtypeEdge, trigtypePulse, etc.)."""
        self._check(
            self._dwf.FDwfAnalogInTriggerTypeSet(self._hdwf, trig_type),
            "FDwfAnalogInTriggerTypeSet",
        )

    def analog_in_set_trigger_level(self, level_v: float) -> None:
        """Set the trigger voltage level in volts."""
        self._check(
            self._dwf.FDwfAnalogInTriggerLevelSet(
                self._hdwf, ctypes.c_double(level_v)
            ),
            "FDwfAnalogInTriggerLevelSet",
        )

    def analog_in_set_trigger_condition(
        self, condition: int = DwfTriggerSlopeRise
    ) -> None:
        """Set trigger slope: DwfTriggerSlopeRise, DwfTriggerSlopeFall, DwfTriggerSlopeEither."""
        self._check(
            self._dwf.FDwfAnalogInTriggerConditionSet(self._hdwf, condition),
            "FDwfAnalogInTriggerConditionSet",
        )

    def analog_in_set_trigger_hysteresis(self, hysteresis_v: float) -> None:
        """Set trigger hysteresis in volts."""
        self._check(
            self._dwf.FDwfAnalogInTriggerHysteresisSet(
                self._hdwf, ctypes.c_double(hysteresis_v)
            ),
            "FDwfAnalogInTriggerHysteresisSet",
        )

    def analog_in_set_trigger_auto_timeout(self, timeout_s: float) -> None:
        """
        Set auto-trigger timeout in seconds.
        0 = normal trigger (no auto-fire); >0 = auto-trigger after timeout.
        """
        self._check(
            self._dwf.FDwfAnalogInTriggerAutoTimeoutSet(
                self._hdwf, ctypes.c_double(timeout_s)
            ),
            "FDwfAnalogInTriggerAutoTimeoutSet",
        )

    def analog_in_set_trigger_position(self, position_s: float) -> None:
        """
        Set horizontal trigger position in seconds.
        Relative to the buffer midpoint for Single mode, relative to start for Record mode.
        """
        self._check(
            self._dwf.FDwfAnalogInTriggerPositionSet(
                self._hdwf, ctypes.c_double(position_s)
            ),
            "FDwfAnalogInTriggerPositionSet",
        )

    def analog_in_force_trigger(self) -> None:
        """Immediately force-trigger the Analog In acquisition."""
        self._check(
            self._dwf.FDwfAnalogInTriggerForce(self._hdwf),
            "FDwfAnalogInTriggerForce",
        )

    # --- Data retrieval -------------------------------------------------

    def analog_in_read_sample(self, channel: int) -> float:
        """
        Read a single instantaneous ADC sample from *channel* (0-based).
        Does not require a full acquisition cycle.
        """
        self._dwf.FDwfAnalogInConfigure(self._hdwf, 0, 0)  # no-op reconfigure
        v = ctypes.c_double(0.0)
        self._check(
            self._dwf.FDwfAnalogInStatusSample(self._hdwf, channel, ctypes.byref(v)),
            "FDwfAnalogInStatusSample",
        )
        return v.value

    def analog_in_get_data(self, channel: int, n_samples: int) -> np.ndarray:
        """
        Read *n_samples* voltage samples from *channel* after a status update.
        Call ``analog_in_status(read_data=True)`` first.
        """
        buf = (ctypes.c_double * n_samples)()
        self._check(
            self._dwf.FDwfAnalogInStatusData(self._hdwf, channel, buf, n_samples),
            "FDwfAnalogInStatusData",
        )
        return np.frombuffer(buf, dtype=np.float64).copy()

    def analog_in_capture(
        self,
        channel: int = 0,
        sample_rate_hz: float = 1e6,
        buffer_size: int = 4096,
        y_offset: float = 0.0,
        y_range: float = 5.0,
        attenuation: float = 1.0,
        trigger_level_v: Optional[float] = None,
        trigger_channel: Optional[int] = None,
        trigger_condition: int = DwfTriggerSlopeRise,
        auto_timeout_s: float = 1.0,
        timeout_s: float = 5.0,
    ) -> np.ndarray:
        """
        Perform a single triggered (or auto-triggered) capture on *channel*.

        Parameters
        ----------
        channel : int
            Zero-based channel index to acquire.
        sample_rate_hz : float
            ADC sample rate in Hz.
        buffer_size : int
            Number of samples to capture.
        y_offset : float
            Offset to channel (V).
        y_range : float
            Pk to pk voltage range (also defines channel resolution) (V).
        attenuation : float
            Attenuation of channel (effectively multiplier) (unitless).
        trigger_level_v : float or None
            Trigger voltage level.  If None, no hardware trigger is set
            (free-run / auto-trigger only).
        trigger_channel : int or None
            Channel to trigger on.  Defaults to *channel*.
        trigger_condition : int
            DwfTriggerSlopeRise / DwfTriggerSlopeFall / DwfTriggerSlopeEither.
        auto_timeout_s : float
            Auto-trigger timeout (seconds).  Use 0 for strict "Normal" trigger.
        timeout_s : float
            Host-side acquisition timeout before raising TimeoutError.

        Returns
        -------
        np.ndarray
            Array of ``buffer_size`` voltage samples (float64, volts).
        """
        self.analog_in_reset()
        self.analog_in_set_sample_rate(sample_rate_hz)
        self.analog_in_set_buffer_size(buffer_size)
        self.analog_in_set_acquisition_mode(acqmodeSingle)
        self.analog_in_set_offset(channel, y_offset)
        self.analog_in_set_range(channel, y_range)
        self.analog_in_set_attenuation(channel, attenuation)
        self.analog_in_channel_enable(channel)

        if trigger_level_v is not None:
            trig_ch = channel if trigger_channel is None else trigger_channel
            self.analog_in_set_trigger_source(trigsrcDetectorAnalogIn)
            self.analog_in_set_trigger_channel(trig_ch)
            self.analog_in_set_trigger_type(trigtypeEdge)
            self.analog_in_set_trigger_level(trigger_level_v)
            self.analog_in_set_trigger_condition(trigger_condition)
            self.analog_in_set_trigger_auto_timeout(auto_timeout_s)
        else:
            self.analog_in_set_trigger_source(trigsrcNone)

        self.analog_in_configure(reconfigure=True, start=True)

        deadline = time.time() + timeout_s
        while True:
            state = self.analog_in_status(read_data=True)
            if state == DwfStateDone:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"analog_in_capture: acquisition did not complete within "
                    f"{timeout_s:.1f} s (last state = {self._state_name(state)})"
                )
            time.sleep(0.001)

        return self.analog_in_get_data(channel, buffer_size)
    
    def analog_in_capture_multiple(
        self,
        channel_settings: dict,
        sample_rate_hz: float = 1e6,
        buffer_size: int = 4096,
        trigger_level_v: Optional[float] = None,
        trigger_channel: Optional[int] = None,
        trigger_condition: int = DwfTriggerSlopeRise,
        auto_timeout_s: float = 1.0,
        timeout_s: float = 5.0,
    ) -> dict:
        """More generalized version of analog in capture for easy settings for multiple channels.

        Args:
            channel_settings (dict): 
                Dictionary where keys are channel numbers (0 indexed) and values are 
                dictionaries of parameters (y_range, y_offset, attenuation).
            sample_rate_hz : float
                ADC sample rate in Hz.
            buffer_size : int
                Number of samples to capture.
            trigger_level_v : float or None
                Trigger voltage level.  If None, no hardware trigger is set
                (free-run / auto-trigger only).
            trigger_channel : int or None
                Channel to trigger on.  Defaults to *channel*.
            trigger_condition : int
                DwfTriggerSlopeRise / DwfTriggerSlopeFall / DwfTriggerSlopeEither.
            auto_timeout_s : float
                Auto-trigger timeout (seconds).  Use 0 for strict "Normal" trigger.
            timeout_s : float
                Host-side acquisition timeout before raising TimeoutError.

        Returns:
            dict:
                Dictionary where keys are channel numbers (0 indexed) and values are
                np.ndarray of ``buffer_size`` voltage samples (float64, volts)
        """
        if trigger_level_v is not None and trigger_channel is None:
            raise TriggerWithoutChannelError

        self.analog_in_reset()
        self.analog_in_set_sample_rate(sample_rate_hz)
        self.analog_in_set_buffer_size(buffer_size)
        self.analog_in_set_acquisition_mode(acqmodeSingle)

        channels = channel_settings.keys()

        for channel in channels:
            self.analog_in_channel_enable(channel)
            y_offset, y_range, attenuation = channels[channel]["y_offset"], channels[channel]["y_range"], channels[channel]["attenuation"]
            self.analog_in_set_offset(channel, y_offset)
            self.analog_in_set_range(channel, y_range)
            self.analog_in_set_attenuation(channel, attenuation)

        if trigger_level_v is not None:
            self.analog_in_set_trigger_source(trigsrcDetectorAnalogIn)
            self.analog_in_set_trigger_channel(trigger_channel)
            self.analog_in_set_trigger_type(trigtypeEdge)
            self.analog_in_set_trigger_level(trigger_level_v)
            self.analog_in_set_trigger_condition(trigger_condition)
            self.analog_in_set_trigger_auto_timeout(auto_timeout_s)
        else:
            self.analog_in_set_trigger_source(trigsrcNone)

        self.analog_in_configure(reconfigure=True, start=True)

        deadline = time.time() + timeout_s
        while True:
            state = self.analog_in_status(read_data=True)
            if state == DwfStateDone:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"analog_in_capture: acquisition did not complete within "
                    f"{timeout_s:.1f} s (last state = {self._state_name(state)})"
                )
            time.sleep(0.001)
        
        results = {}
        for channel in channels:
            results[channel] = self.analog_in_get_data(channel, buffer_size)

        return results



    def analog_in_record(
        self,
        channel: int = 0,
        sample_rate_hz: float = 1e6,
        record_length_s: float = 1.0,
        timeout_s: Optional[float] = None,
    ) -> np.ndarray:
        """
        Stream a long recording (acqmodeRecord) from *channel*.

        Parameters
        ----------
        channel : int
            Zero-based channel to record.
        sample_rate_hz : float
            ADC sample rate in Hz.
        record_length_s : float
            Recording duration in seconds.
        timeout_s : float or None
            Host-side timeout.  Defaults to ``record_length_s * 3``.

        Returns
        -------
        np.ndarray  - float64 voltage samples (may be shorter than requested
                      if data loss occurred; a warning is printed).
        """
        if timeout_s is None:
            timeout_s = record_length_s * 3 + 5.0

        self.analog_in_reset()
        self.analog_in_set_sample_rate(sample_rate_hz)
        self.analog_in_set_acquisition_mode(acqmodeRecord)
        self.analog_in_channel_enable(channel)

        self._check(
            self._dwf.FDwfAnalogInRecordLengthSet(
                self._hdwf, ctypes.c_double(record_length_s)
            ),
            "FDwfAnalogInRecordLengthSet",
        )
        self.analog_in_configure(reconfigure=True, start=True)

        n_total = int(sample_rate_hz * record_length_s)
        samples: List[np.ndarray] = []
        total_received = 0
        total_lost = 0
        deadline = time.time() + timeout_s

        while total_received < n_total:
            state = self.analog_in_status(read_data=True)

            avail   = ctypes.c_int(0)
            lost    = ctypes.c_int(0)
            corrupt = ctypes.c_int(0)
            self._dwf.FDwfAnalogInStatusRecord(
                self._hdwf,
                ctypes.byref(avail),
                ctypes.byref(lost),
                ctypes.byref(corrupt),
            )
            total_lost += lost.value

            if avail.value > 0:
                chunk = (ctypes.c_double * avail.value)()
                self._dwf.FDwfAnalogInStatusData2(
                    self._hdwf, channel, chunk, total_received, avail.value
                )
                samples.append(np.frombuffer(chunk, dtype=np.float64).copy())
                total_received += avail.value

            if state == DwfStateDone and avail.value == 0:
                break
            if time.time() > deadline:
                raise TimeoutError("analog_in_record: timed out.")
            time.sleep(0.001)

        if total_lost:
            print(f"Warning: {total_lost} samples lost during recording.")
        return np.concatenate(samples) if samples else np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # Analog Out – Arbitrary Waveform Generator
    # ------------------------------------------------------------------

    def analog_out_channel_count(self) -> int:
        """Return the number of Analog Out channels."""
        n = ctypes.c_int(0)
        self._check(
            self._dwf.FDwfAnalogOutCount(self._hdwf, ctypes.byref(n)),
            "FDwfAnalogOutCount",
        )
        return n.value

    def analog_out_reset(self, channel: int = -1) -> None:
        """Reset Analog Out channel(s).  Use channel=-1 to reset all."""
        self._check(
            self._dwf.FDwfAnalogOutReset(self._hdwf, channel),
            "FDwfAnalogOutReset",
        )

    def analog_out_start(self, channel: int = 0) -> None:
        """Start the Analog Out generator on *channel*."""
        self._check(
            self._dwf.FDwfAnalogOutConfigure(self._hdwf, channel, 1),
            "FDwfAnalogOutConfigure(start)",
        )

    def analog_out_stop(self, channel: int = 0) -> None:
        """Stop the Analog Out generator on *channel*."""
        self._check(
            self._dwf.FDwfAnalogOutConfigure(self._hdwf, channel, 0),
            "FDwfAnalogOutConfigure(stop)",
        )

    def analog_out_status(self, channel: int) -> int:
        """Return the current DwfState of the Analog Out channel."""
        sts = ctypes.c_byte(0)
        self._check(
            self._dwf.FDwfAnalogOutStatus(self._hdwf, channel, ctypes.byref(sts)),
            "FDwfAnalogOutStatus",
        )
        return int(sts.value)

    # --- Carrier node configuration ------------------------------------

    def analog_out_set_function(
        self,
        channel: int,
        func: int = funcSine,
        node: int = AnalogOutNodeCarrier,
    ) -> None:
        """Set the waveform function (funcSine, funcSquare, funcCustom, etc.)."""
        self._check(
            self._dwf.FDwfAnalogOutNodeFunctionSet(self._hdwf, channel, node, func),
            "FDwfAnalogOutNodeFunctionSet",
        )

    def analog_out_set_frequency(
        self,
        channel: int,
        freq_hz: float,
        node: int = AnalogOutNodeCarrier,
    ) -> None:
        """Set the output frequency in Hz."""
        self._check(
            self._dwf.FDwfAnalogOutNodeFrequencySet(
                self._hdwf, channel, node, ctypes.c_double(freq_hz)
            ),
            "FDwfAnalogOutNodeFrequencySet",
        )

    def analog_out_set_amplitude(
        self,
        channel: int,
        amplitude_v: float,
        node: int = AnalogOutNodeCarrier,
    ) -> None:
        """Set the output amplitude in volts (peak value, not peak-to-peak)."""
        self._check(
            self._dwf.FDwfAnalogOutNodeAmplitudeSet(
                self._hdwf, channel, node, ctypes.c_double(amplitude_v)
            ),
            "FDwfAnalogOutNodeAmplitudeSet",
        )

    def analog_out_set_offset(
        self,
        channel: int,
        offset_v: float,
        node: int = AnalogOutNodeCarrier,
    ) -> None:
        """Set the DC offset in volts."""
        self._check(
            self._dwf.FDwfAnalogOutNodeOffsetSet(
                self._hdwf, channel, node, ctypes.c_double(offset_v)
            ),
            "FDwfAnalogOutNodeOffsetSet",
        )

    def analog_out_set_phase(
        self,
        channel: int,
        phase_deg: float,
        node: int = AnalogOutNodeCarrier,
    ) -> None:
        """Set the output phase in degrees (0–360)."""
        self._check(
            self._dwf.FDwfAnalogOutNodePhaseSet(
                self._hdwf, channel, node, ctypes.c_double(phase_deg)
            ),
            "FDwfAnalogOutNodePhaseSet",
        )

    def analog_out_set_symmetry(
        self,
        channel: int,
        symmetry_pct: float,
        node: int = AnalogOutNodeCarrier,
    ) -> None:
        """
        Set waveform symmetry / duty-cycle percentage (0–100).
        For square waves this controls duty cycle; for triangle it controls ramp skew.
        """
        self._check(
            self._dwf.FDwfAnalogOutNodeSymmetrySet(
                self._hdwf, channel, node, ctypes.c_double(symmetry_pct)
            ),
            "FDwfAnalogOutNodeSymmetrySet",
        )

    def analog_out_enable_node(
        self,
        channel: int,
        node: int = AnalogOutNodeCarrier,
        enable: int = 1,
    ) -> None:
        """Enable or disable an output node (0 = disable, 1 = enable)."""
        self._check(
            self._dwf.FDwfAnalogOutNodeEnableSet(self._hdwf, channel, node, enable),
            "FDwfAnalogOutNodeEnableSet",
        )

    # --- Convenience waveform generators --------------------------------

    def analog_out_set_sine(
        self,
        channel: int,
        freq_hz: float,
        amplitude_v: float = 1.0,
        offset_v: float = 0.0,
        phase_deg: float = 0.0,
    ) -> None:
        """Configure *channel* to output a sine wave (does not start)."""
        self.analog_out_reset(channel)
        self.analog_out_enable_node(channel, AnalogOutNodeCarrier, 1)
        self.analog_out_set_function(channel, funcSine)
        self.analog_out_set_frequency(channel, freq_hz)
        self.analog_out_set_amplitude(channel, amplitude_v)
        self.analog_out_set_offset(channel, offset_v)
        self.analog_out_set_phase(channel, phase_deg)

    def analog_out_set_square(
        self,
        channel: int,
        freq_hz: float,
        amplitude_v: float = 1.0,
        offset_v: float = 0.0,
        duty_cycle_pct: float = 50.0,
    ) -> None:
        """Configure *channel* to output a square / PWM wave (does not start)."""
        self.analog_out_reset(channel)
        self.analog_out_enable_node(channel, AnalogOutNodeCarrier, 1)
        self.analog_out_set_function(channel, funcSquare)
        self.analog_out_set_frequency(channel, freq_hz)
        self.analog_out_set_amplitude(channel, amplitude_v)
        self.analog_out_set_offset(channel, offset_v)
        self.analog_out_set_symmetry(channel, duty_cycle_pct)

    def analog_out_set_dc(
        self,
        channel: int,
        voltage_v: float,
    ) -> None:
        """Configure *channel* to output a constant DC voltage (does not start)."""
        self.analog_out_reset(channel)
        self.analog_out_enable_node(channel, AnalogOutNodeCarrier, 1)
        self.analog_out_set_function(channel, funcDC)
        self.analog_out_set_offset(channel, voltage_v)

    def analog_out_set_custom(
        self,
        channel: int,
        samples: np.ndarray,
        freq_hz: float,
        amplitude_v: float = 1.0,
        offset_v: float = 0.0,
    ) -> None:
        """
        Configure *channel* to output an arbitrary waveform (does not start).

        Parameters
        ----------
        samples : np.ndarray
            Normalised waveform samples in the range [-1, +1].
        freq_hz : float
            Repetition frequency of the waveform in Hz.
        amplitude_v : float
            Peak amplitude in volts.  Output = offset ± amplitude * sample.
        offset_v : float
            DC offset in volts.
        """
        samples = np.asarray(samples, dtype=np.float64)
        if samples.max() > 1.0 or samples.min() < -1.0:
            raise ValueError("Custom waveform samples must be normalised to [-1, 1].")
        self.analog_out_reset(channel)
        self.analog_out_enable_node(channel, AnalogOutNodeCarrier, 1)
        self.analog_out_set_function(channel, funcCustom)
        self.analog_out_set_frequency(channel, freq_hz)
        self.analog_out_set_amplitude(channel, amplitude_v)
        self.analog_out_set_offset(channel, offset_v)
        c_buf = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._check(
            self._dwf.FDwfAnalogOutNodeDataSet(
                self._hdwf, channel, AnalogOutNodeCarrier, c_buf, len(samples)
            ),
            "FDwfAnalogOutNodeDataSet",
        )

    # --- Run / Wait / Repeat settings -----------------------------------

    def analog_out_set_run_time(self, channel: int, seconds: float) -> None:
        """Set how long the generator runs before entering Done state (0 = infinite)."""
        self._check(
            self._dwf.FDwfAnalogOutRunSet(
                self._hdwf, channel, ctypes.c_double(seconds)
            ),
            "FDwfAnalogOutRunSet",
        )

    def analog_out_set_repeat(self, channel: int, count: int) -> None:
        """Set the number of repetitions (0 = infinite)."""
        self._check(
            self._dwf.FDwfAnalogOutRepeatSet(self._hdwf, channel, count),
            "FDwfAnalogOutRepeatSet",
        )

    def analog_out_set_trigger_source(
        self, channel: int, source: int = trigsrcNone
    ) -> None:
        """Set the Analog Out trigger source."""
        self._check(
            self._dwf.FDwfAnalogOutTriggerSourceSet(self._hdwf, channel, source),
            "FDwfAnalogOutTriggerSourceSet",
        )

    # ------------------------------------------------------------------
    # Analog I/O – Power supplies, system monitor
    # ------------------------------------------------------------------

    def analog_io_enable(self, enable: bool = True) -> None:
        """Enable or disable Analog I/O master switch."""
        self._check(
            self._dwf.FDwfAnalogIOEnableSet(self._hdwf, int(enable)),
            "FDwfAnalogIOEnableSet",
        )

    def analog_io_status(self) -> None:
        """Read the Analog I/O status (required before reading node values)."""
        self._check(
            self._dwf.FDwfAnalogIOStatus(self._hdwf),
            "FDwfAnalogIOStatus",
        )

    def analog_io_channel_node_set(
        self, channel: int, node: int, value: float
    ) -> None:
        """Set the value of an Analog I/O channel node (e.g. supply voltage)."""
        self._check(
            self._dwf.FDwfAnalogIOChannelNodeSet(
                self._hdwf, channel, node, ctypes.c_double(value)
            ),
            "FDwfAnalogIOChannelNodeSet",
        )

    def analog_io_channel_node_get(self, channel: int, node: int) -> float:
        """Return the configured value of an Analog I/O channel node."""
        v = ctypes.c_double(0.0)
        self._check(
            self._dwf.FDwfAnalogIOChannelNodeGet(
                self._hdwf, channel, node, ctypes.byref(v)
            ),
            "FDwfAnalogIOChannelNodeGet",
        )
        return v.value

    def analog_io_channel_node_status(self, channel: int, node: int) -> float:
        """Return the live measured value of an Analog I/O channel node."""
        v = ctypes.c_double(0.0)
        self._check(
            self._dwf.FDwfAnalogIOChannelNodeStatus(
                self._hdwf, channel, node, ctypes.byref(v)
            ),
            "FDwfAnalogIOChannelNodeStatus",
        )
        return v.value

    # Convenience: Analog Discovery positive/negative power supplies
    # Channel 0 = V+, Channel 1 = V- (device-specific, check your model)

    def power_supply_set(
        self,
        positive_v: float = 3.3,
        negative_v: float = -3.3,
    ) -> None:
        """
        Configure and enable the Analog Discovery ± power supplies.

        Node 0 on each channel is the voltage setpoint; node 1 is the
        enable flag.  This is the common layout for AD2 / AD3.
        Consult your device's channel-node map if outputs differ.
        """
        # Positive supply: channel 0, node 0 = voltage, node 1 = enable
        self.analog_io_channel_node_set(0, 0, positive_v)
        self.analog_io_channel_node_set(0, 1, 1.0)
        # Negative supply: channel 1, node 0 = voltage, node 1 = enable
        self.analog_io_channel_node_set(1, 0, negative_v)
        self.analog_io_channel_node_set(1, 1, 1.0)
        self.analog_io_enable(True)

    def power_supply_off(self) -> None:
        """Disable the ± power supplies."""
        self.analog_io_channel_node_set(0, 1, 0.0)
        self.analog_io_channel_node_set(1, 1, 0.0)
        self.analog_io_enable(False)

    # ------------------------------------------------------------------
    # Digital I/O
    # ------------------------------------------------------------------

    def digital_io_reset(self) -> None:
        """Reset the Digital I/O instrument."""
        self._check(
            self._dwf.FDwfDigitalIOReset(self._hdwf),
            "FDwfDigitalIOReset",
        )

    def digital_io_status(self) -> None:
        """Read back Digital I/O state (required before reading pin states)."""
        self._check(
            self._dwf.FDwfDigitalIOStatus(self._hdwf),
            "FDwfDigitalIOStatus",
        )

    def digital_io_set_output_enable(self, pin_mask: int) -> None:
        """
        Configure pins as outputs using a bitmask.
        Example: ``0x00FF`` makes pins 0–7 outputs, the rest inputs.
        """
        self._check(
            self._dwf.FDwfDigitalIOOutputEnableSet(self._hdwf, pin_mask),
            "FDwfDigitalIOOutputEnableSet",
        )

    def digital_io_set_output(self, pin_mask: int) -> None:
        """
        Set output pin levels using a bitmask.
        Example: ``0x0001`` sets pin 0 high, all others low.
        """
        self._check(
            self._dwf.FDwfDigitalIOOutputSet(self._hdwf, pin_mask),
            "FDwfDigitalIOOutputSet",
        )

    def digital_io_get_input(self) -> int:
        """
        Return current logic levels of all digital I/O pins as a bitmask.
        Call ``digital_io_status()`` first to refresh the reading.
        """
        mask = ctypes.c_uint32(0)
        self._check(
            self._dwf.FDwfDigitalIOInputStatus(self._hdwf, ctypes.byref(mask)),
            "FDwfDigitalIOInputStatus",
        )
        return mask.value

    def digital_io_read_pin(self, pin: int) -> bool:
        """
        Read the logic level of a single digital I/O pin (0-based index).
        Calls ``digital_io_status()`` internally.
        """
        self.digital_io_status()
        return bool((self.digital_io_get_input() >> pin) & 1)

    def digital_io_write_pin(self, pin: int, value: bool) -> None:
        """
        Set a single digital output pin high (True) or low (False).
        The pin must already be configured as an output.
        """
        # Read–modify–write the current output register
        cur = ctypes.c_uint32(0)
        self._dwf.FDwfDigitalIOOutputGet(self._hdwf, ctypes.byref(cur))
        if value:
            new_mask = cur.value | (1 << pin)
        else:
            new_mask = cur.value & ~(1 << pin)
        self.digital_io_set_output(new_mask)

    # ------------------------------------------------------------------
    # Trigger bus helpers
    # ------------------------------------------------------------------

    def trigger_pc_pulse(self) -> None:
        """Generate a software trigger pulse on the PC trigger line."""
        self._check(
            self._dwf.FDwfDeviceTriggerPC(self._hdwf),
            "FDwfDeviceTriggerPC",
        )

    def trigger_set_pin(self, pin_index: int, source: int) -> None:
        """Configure a trigger I/O pin with the given TRIGSRC source."""
        self._check(
            self._dwf.FDwfDeviceTriggerSet(self._hdwf, pin_index, source),
            "FDwfDeviceTriggerSet",
        )

    # ------------------------------------------------------------------
    # Utility / context helpers
    # ------------------------------------------------------------------

    @contextmanager
    def outputs_enabled(self):
        """Context manager that enables device outputs on enter and disables on exit."""
        self._dwf.FDwfDeviceEnableSet(self._hdwf, 1)
        try:
            yield self
        finally:
            self._dwf.FDwfDeviceEnableSet(self._hdwf, 0)
    def __repr__(self) -> str:
        state = "open" if self._hdwf.value != 0 else "closed"
        return f"<WaveFormsADS handle={self._hdwf.value} [{state}] dwf={self.get_version() if state == 'open' else '?'}>"


# ---------------------------------------------------------------------------
# Quick demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Connected devices:")
    for d in WaveFormsADS.enumerate():
        print(f"  [{d['index']}] {d['name']}  SN:{d['serial']}  open:{d['is_open']}")

    with WaveFormsADS() as dev:
        print(f"\nOpened: {dev}")
        print(f"  Analog In channels  : {dev.analog_in_channel_count()}")
        print(f"  Analog Out channels : {dev.analog_out_channel_count()}")

        # Single sample read
        v = dev.analog_in_read_sample(channel=0)
        print(f"  Ch0 instant sample  : {v:.4f} V")

        # Generate 1 kHz sine on AWG Ch0 for 1 second
        dev.analog_out_set_sine(channel=0, freq_hz=1_000, amplitude_v=1.0)
        dev.analog_out_start(channel=0)
        print("  Outputting 1 kHz sine on AWG Ch0 …")
        time.sleep(1.0)
        dev.analog_out_stop(channel=0)
        print("  Done.")