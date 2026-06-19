"""
ttl_trigger.py
==============
Hardware-timed TTL trigger output for the Digilent Analog Discovery Series
(ADS) using the WaveForms **Digital Out** (pattern generator) instrument.

Why Digital Out instead of Digital I/O static pins?
----------------------------------------------------
``FDwfDigitalIO*`` (used in ``WaveFormsADS.digital_io_write_pin``) is a
simple, software-driven register write.  The pulse actually appears on the
pin whenever Python calls into the DWF library — subject to OS scheduling
jitter of hundreds of microseconds to several milliseconds.

``FDwfDigitalOut*`` (used here) is the **on-device pattern generator**.
Once configured and started, all timing runs inside the AD2/ADP3450
FPGA fabric.  Pulse widths and inter-pulse delays are accurate to the
device clock period (~10 ns on AD2, ~4 ns on ADP3450) with no host-side
jitter in the critical path.

Architecture
------------
``TTLTriggerConfig``   — dataclass describing a single-pin or multi-pin
                         trigger pulse train.
``TTLTrigger``         — high-level controller; wraps a ``WaveFormsADS``
                         instance and drives ``FDwfDigitalOut`` calls.

Quick start
-----------
    from waveforms_ads import WaveFormsADS
    from ttl_trigger import TTLTrigger, TTLTriggerConfig, TTLIdleState

    cfg = TTLTriggerConfig(
        pins        = [0],          # DIO pin 0
        high_time_s = 10e-6,        # 10 µs pulse width
        low_time_s  = 990e-6,       # 990 µs between pulses → 1 kHz
        pulse_count = 5,            # fire 5 pulses then stop  (0 = continuous)
        idle_state  = TTLIdleState.LOW,
        delay_s     = 0.0,          # no pre-trigger delay
    )

    with WaveFormsADS() as dev:
        trig = TTLTrigger(dev)
        trig.configure(cfg)
        trig.fire()                 # non-blocking: hardware fires immediately
        trig.wait_until_done()      # optional: block until all pulses sent
        trig.stop()

Notes on voltage levels
-----------------------
Analog Discovery 2 / ADP3450 Digital Out pins are **3.3 V CMOS** outputs
(IOL / IOH ~±2 mA).  They are TTL-compatible inputs to virtually all
Allied Vision cameras.  Do not drive loads > 4 mA without a buffer.
The ``amplitude_v`` parameter below configures the supply to the digital
output bank (if the device supports it); leave it at 3.3 for standard TTL.
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from waveforms_ads import DWFError, WaveFormsADS

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DWF Digital-Out constants not already in waveforms_ads.py
# ---------------------------------------------------------------------------

# DigitalOut idle states
DwfDigitalOutIdleInit    = 0   # return to initial value
DwfDigitalOutIdleLow     = 1   # hold low
DwfDigitalOutIdleHigh    = 2   # hold high
DwfDigitalOutIdleZet     = 3   # tri-state (high-Z)

# DigitalOut output types
DwfDigitalOutTypePulse   = 0   # hardware pulse generator
DwfDigitalOutTypeCustom  = 1   # arbitrary pattern from buffer
DwfDigitalOutTypeRandom  = 2   # PRBS / random

# DigitalOut trigger sources (reuse analog constants where they overlap)
trigsrcNone    = 0
trigsrcPC      = 1             # software trigger via FDwfDeviceTriggerPC


# ---------------------------------------------------------------------------
# Public enumerations
# ---------------------------------------------------------------------------

class TTLIdleState(Enum):
    """Logic level of the pin when no pulse is being generated."""
    LOW      = DwfDigitalOutIdleLow
    HIGH     = DwfDigitalOutIdleHigh
    TRISTATE = DwfDigitalOutIdleZet


class TTLPolarity(Enum):
    """
    Polarity of the active (triggered) pulse.

    ACTIVE_HIGH: pin is LOW at idle, goes HIGH for ``high_time_s``.
    ACTIVE_LOW:  pin is HIGH at idle, goes LOW for ``high_time_s``.

    Note: ``TTLIdleState`` takes precedence for the between-pulse resting
    level.  ``TTLPolarity`` only affects which direction is the "pulse".
    """
    ACTIVE_HIGH = "active_high"
    ACTIVE_LOW  = "active_low"


class TTLTriggerMode(Enum):
    """
    When to start generating pulses after ``fire()`` is called.

    IMMEDIATE  — output starts as soon as the device receives the configure
                 command (lowest software latency path, but still subject to
                 USB transfer time, ~1-2 ms).
    SOFTWARE   — output waits for an explicit ``FDwfDeviceTriggerPC`` call
                 (``trig.fire_software_trigger()``), allowing you to arm
                 the output in advance and fire it with a single fast call.
    """
    IMMEDIATE = "immediate"
    SOFTWARE  = "software"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TTLTriggerConfig:
    """
    Full specification for a TTL pulse train on one or more Digital Out pins.

    Parameters
    ----------
    pins : list[int]
        One or more zero-based DIO pin indices to drive simultaneously.
        All listed pins receive identical timing.
        Analog Discovery 2 has pins 0-15; ADP3450 has 0-31.
    high_time_s : float
        Duration the pin is in its active (high or low, per polarity) state,
        in seconds.  Minimum is one device-clock period (~10 ns on AD2).
    low_time_s : float
        Duration the pin is in its idle state between pulses, in seconds.
        Together with ``high_time_s`` this defines the period:
        ``period = high_time_s + low_time_s``.
    pulse_count : int
        Number of pulses to generate.  0 = run continuously until ``stop()``.
    idle_state : TTLIdleState
        Logic level of the pin(s) when not pulsing (before first pulse and
        after last pulse).
    polarity : TTLPolarity
        Direction of the active pulse edge.
    mode : TTLTriggerMode
        IMMEDIATE (start on ``fire()``) or SOFTWARE (wait for
        ``fire_software_trigger()`` after ``fire()``).
    delay_s : float
        Pre-trigger delay between the start command and the first pulse edge,
        in seconds.  Implemented as an initial low/high phase using the
        device's counter, so it is also hardware-timed.
    amplitude_v : float
        Output voltage for digital HIGH.  On devices that expose a
        ``DigitalVoltageSet`` feature (e.g. ADP3450), this sets the IO bank
        voltage.  Ignored silently on AD2 (fixed 3.3 V).  Use 3.3 for
        standard TTL compatibility.
    """
    pins:        List[int]    = field(default_factory=lambda: [0])
    high_time_s: float        = 10e-6      # 10 µs pulse width
    low_time_s:  float        = 990e-6     # 990 µs gap  → 1 kHz at 1% duty
    pulse_count: int          = 1          # 0 = continuous
    idle_state:  TTLIdleState = TTLIdleState.LOW
    polarity:    TTLPolarity  = TTLPolarity.ACTIVE_HIGH
    mode:        TTLTriggerMode = TTLTriggerMode.IMMEDIATE
    delay_s:     float        = 0.0
    amplitude_v: float        = 3.3

    # --- Derived / read-only properties ----------------------------------

    @property
    def period_s(self) -> float:
        """Total period of one pulse cycle in seconds."""
        return self.high_time_s + self.low_time_s

    @property
    def frequency_hz(self) -> float:
        """Repetition frequency in Hz."""
        return 1.0 / self.period_s if self.period_s > 0 else float("inf")

    @property
    def duty_cycle_pct(self) -> float:
        """Active-phase duty cycle as a percentage (0–100)."""
        return 100.0 * self.high_time_s / self.period_s if self.period_s > 0 else 0.0

    def __post_init__(self) -> None:
        if not self.pins:
            raise ValueError("pins must contain at least one pin index.")
        if self.high_time_s <= 0:
            raise ValueError("high_time_s must be positive.")
        if self.low_time_s < 0:
            raise ValueError("low_time_s must be non-negative.")
        if self.pulse_count < 0:
            raise ValueError("pulse_count must be >= 0 (0 = continuous).")
        if self.delay_s < 0:
            raise ValueError("delay_s must be non-negative.")


# ---------------------------------------------------------------------------
# TTL trigger controller
# ---------------------------------------------------------------------------

class TTLTrigger:
    """
    Controls the Digital Out (pattern generator) instrument on a Digilent
    Analog Discovery Series device to produce hardware-timed TTL pulses.

    All timing is executed by the device FPGA — once ``fire()`` or
    ``fire_software_trigger()`` is called, no further Python involvement is
    needed and pulse edges are accurate to ~10 ns (AD2) / ~4 ns (ADP3450).

    Parameters
    ----------
    device : WaveFormsADS
        An open ``WaveFormsADS`` instance.  The ``TTLTrigger`` does not own
        the device and will not close it.

    Example
    -------
    >>> with WaveFormsADS() as dev:
    ...     trig = TTLTrigger(dev)
    ...     cfg = TTLTriggerConfig(pins=[0, 1], high_time_s=50e-6,
    ...                            low_time_s=950e-6, pulse_count=10)
    ...     trig.configure(cfg)
    ...     trig.fire()
    ...     trig.wait_until_done(timeout_s=5.0)
    """

    # The DWF Digital Out clock is 100 MHz on AD2 (10 ns resolution).
    # ADP3450 runs at 250 MHz (4 ns).  We read the actual rate at configure
    # time rather than hard-coding it.
    _FALLBACK_CLOCK_HZ = 100_000_000.0

    def __init__(self, device: WaveFormsADS) -> None:
        self._dev = device
        self._dwf = device._dwf
        self._hdwf = device._hdwf
        self._cfg: Optional[TTLTriggerConfig] = None
        self._clock_hz: float = self._FALLBACK_CLOCK_HZ
        self._configured = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(self, cfg: TTLTriggerConfig) -> None:
        """
        Program the Digital Out instrument with the given ``TTLTriggerConfig``.

        This call does **not** start the output — call ``fire()`` or
        ``fire_software_trigger()`` afterwards.

        The method can be called repeatedly to update timing without stopping
        and re-starting the device (reconfigure-while-armed).

        Parameters
        ----------
        cfg : TTLTriggerConfig
            Full pulse-train specification.
        """
        self._cfg = cfg

        # 1. Reset the Digital Out instrument
        self._check(
            self._dwf.FDwfDigitalOutReset(self._hdwf),
            "FDwfDigitalOutReset",
        )

        # 2. Read the actual internal clock frequency
        self._clock_hz = self._read_internal_clock()
        _log.debug("Digital Out internal clock: %.0f Hz", self._clock_hz)

        # 3. Optionally set the IO bank voltage (ADP3450 and similar)
        self._try_set_voltage(cfg.amplitude_v)

        # 4. Set the trigger source
        trigger_src = trigsrcPC if cfg.mode == TTLTriggerMode.SOFTWARE else trigsrcNone
        self._check(
            self._dwf.FDwfDigitalOutTriggerSourceSet(self._hdwf, trigger_src),
            "FDwfDigitalOutTriggerSourceSet",
        )

        # 5. Set repeat count (0 = run forever)
        self._check(
            self._dwf.FDwfDigitalOutRepeatSet(self._hdwf, cfg.pulse_count),
            "FDwfDigitalOutRepeatSet",
        )

        # 6. Set run time (how long one "run" lasts before Done)
        #    For finite pulse_count we compute it precisely; for continuous we pass 0.
        if cfg.pulse_count > 0:
            run_time_s = cfg.pulse_count * cfg.period_s + cfg.delay_s
            self._check(
                self._dwf.FDwfDigitalOutRunSet(
                    self._hdwf, ctypes.c_double(run_time_s)
                ),
                "FDwfDigitalOutRunSet",
            )
        else:
            # 0 = run forever
            self._check(
                self._dwf.FDwfDigitalOutRunSet(self._hdwf, ctypes.c_double(0.0)),
                "FDwfDigitalOutRunSet",
            )

        # 7. Configure each pin
        for pin in cfg.pins:
            self._configure_pin(pin, cfg)

        self._configured = True
        _log.info(
            "TTL trigger configured: pins=%s  %.3g µs HIGH / %.3g µs LOW  "
            "(%g Hz, %.1f%% duty)  count=%s  mode=%s  delay=%.3g µs",
            cfg.pins,
            cfg.high_time_s * 1e6,
            cfg.low_time_s * 1e6,
            cfg.frequency_hz,
            cfg.duty_cycle_pct,
            cfg.pulse_count if cfg.pulse_count > 0 else "∞",
            cfg.mode.value,
            cfg.delay_s * 1e6,
        )

    def fire(self) -> None:
        """
        Start the Digital Out instrument.

        For ``TTLTriggerMode.IMMEDIATE``: the first pulse edge appears on
        the pin(s) as soon as the USB command reaches the device (~1–2 ms
        host latency, but then hardware-timed).

        For ``TTLTriggerMode.SOFTWARE``: the instrument arms itself and
        waits for ``fire_software_trigger()``.  Use this pattern when you
        need to minimise the interval between the trigger decision and the
        first edge:

            trig.configure(cfg)       # do early, during setup
            trig.fire()               # arm; no output yet
            # ... wait for the right moment ...
            trig.fire_software_trigger()   # <1 ms to first edge

        Raises
        ------
        RuntimeError
            If ``configure()`` has not been called first.
        """
        if not self._configured:
            raise RuntimeError("Call configure() before fire().")
        self._check(
            self._dwf.FDwfDigitalOutConfigure(self._hdwf, 1),
            "FDwfDigitalOutConfigure(start)",
        )
        _log.info("Digital Out started (%s mode).", self._cfg.mode.value)

    def fire_software_trigger(self) -> None:
        """
        Send the PC software trigger pulse to the device.

        Only meaningful when the config ``mode`` is ``TTLTriggerMode.SOFTWARE``
        and ``fire()`` has already been called (device is armed / waiting).

        This is the fastest way to initiate the output from Python: arm the
        device early, then call this single DWF function at the desired
        moment.  Round-trip USB latency is typically 0.5–2 ms; jitter between
        successive calls is ~10–50 µs.
        """
        self._check(
            self._dwf.FDwfDeviceTriggerPC(self._hdwf),
            "FDwfDeviceTriggerPC",
        )
        _log.debug("Software trigger fired.")

    def stop(self) -> None:
        """Immediately stop the Digital Out instrument and idle all pins."""
        self._check(
            self._dwf.FDwfDigitalOutConfigure(self._hdwf, 0),
            "FDwfDigitalOutConfigure(stop)",
        )
        _log.info("Digital Out stopped.")

    def wait_until_done(self, timeout_s: float = 10.0, poll_interval_s: float = 0.001) -> None:
        """
        Block until the Digital Out instrument reaches the ``Done`` state
        (i.e. all ``pulse_count`` pulses have been sent).

        For continuous mode (``pulse_count=0``) this will block until
        ``timeout_s`` expires — call ``stop()`` explicitly instead.

        Parameters
        ----------
        timeout_s : float
            Maximum time to wait in seconds.
        poll_interval_s : float
            How often to poll the device state.

        Raises
        ------
        TimeoutError
            If the instrument does not reach Done within ``timeout_s``.
        RuntimeError
            If called before ``fire()``.
        """
        DwfStateDone = 2
        deadline = time.monotonic() + timeout_s
        while True:
            sts = ctypes.c_byte(0)
            self._check(
                self._dwf.FDwfDigitalOutStatus(self._hdwf, ctypes.byref(sts)),
                "FDwfDigitalOutStatus",
            )
            if sts.value == DwfStateDone:
                _log.info("Digital Out done.")
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"wait_until_done: Digital Out did not finish within "
                    f"{timeout_s:.1f} s.  "
                    "For continuous mode, call stop() explicitly."
                )
            time.sleep(poll_interval_s)

    def status(self) -> int:
        """Return the current DwfState integer of the Digital Out instrument."""
        sts = ctypes.c_byte(0)
        self._check(
            self._dwf.FDwfDigitalOutStatus(self._hdwf, ctypes.byref(sts)),
            "FDwfDigitalOutStatus",
        )
        return int(sts.value)

    def reset(self) -> None:
        """Reset the Digital Out instrument and clear configuration state."""
        self._check(
            self._dwf.FDwfDigitalOutReset(self._hdwf),
            "FDwfDigitalOutReset",
        )
        self._configured = False
        self._cfg = None
        _log.debug("Digital Out reset.")

    # ------------------------------------------------------------------
    # Convenience: send a single pulse right now
    # ------------------------------------------------------------------

    def send_pulse(
        self,
        pin: int = 0,
        high_time_s: float = 10e-6,
        delay_s: float = 0.0,
        idle_state: TTLIdleState = TTLIdleState.LOW,
        amplitude_v: float = 3.3,
    ) -> None:
        """
        One-shot convenience: configure and immediately fire a single TTL
        pulse on ``pin``, then block until complete.

        Parameters
        ----------
        pin : int
            DIO pin index (0-based).
        high_time_s : float
            Pulse width in seconds.
        delay_s : float
            Pre-pulse delay in seconds.
        idle_state : TTLIdleState
            Resting logic level before and after the pulse.
        amplitude_v : float
            Output high voltage (3.3 V for standard TTL).
        """
        cfg = TTLTriggerConfig(
            pins=[pin],
            high_time_s=high_time_s,
            low_time_s=0.0,          # no repeat gap needed for a single shot
            pulse_count=1,
            idle_state=idle_state,
            polarity=TTLPolarity.ACTIVE_HIGH,
            mode=TTLTriggerMode.IMMEDIATE,
            delay_s=delay_s,
            amplitude_v=amplitude_v,
        )
        self.configure(cfg)
        self.fire()
        self.wait_until_done(timeout_s=delay_s + high_time_s + 1.0)

    def send_pulse_train(
        self,
        pins: List[int],
        high_time_s: float,
        low_time_s: float,
        pulse_count: int,
        idle_state: TTLIdleState = TTLIdleState.LOW,
        polarity: TTLPolarity = TTLPolarity.ACTIVE_HIGH,
        delay_s: float = 0.0,
        amplitude_v: float = 3.3,
        block: bool = True,
        timeout_s: Optional[float] = None,
    ) -> None:
        """
        Convenience wrapper: configure, fire, and optionally wait for a
        complete pulse train on one or more pins.

        Parameters
        ----------
        pins : list[int]
            DIO pin indices (0-based).  All receive identical timing.
        high_time_s : float
            Active pulse width in seconds.
        low_time_s : float
            Idle time between pulses in seconds.
        pulse_count : int
            Number of pulses (0 = continuous until ``stop()``).
        idle_state : TTLIdleState
            Resting level between pulses and after completion.
        polarity : TTLPolarity
            Direction of the active edge.
        delay_s : float
            Pre-trigger delay in seconds.
        amplitude_v : float
            Output high voltage.
        block : bool
            If True, blocks until all pulses have been sent.
        timeout_s : float or None
            Timeout for blocking wait.  Defaults to
            ``pulse_count * period + delay + 2``.
        """
        cfg = TTLTriggerConfig(
            pins=pins,
            high_time_s=high_time_s,
            low_time_s=low_time_s,
            pulse_count=pulse_count,
            idle_state=idle_state,
            polarity=polarity,
            mode=TTLTriggerMode.IMMEDIATE,
            delay_s=delay_s,
            amplitude_v=amplitude_v,
        )
        self.configure(cfg)
        self.fire()
        if block and pulse_count > 0:
            if timeout_s is None:
                timeout_s = pulse_count * cfg.period_s + delay_s + 2.0
            self.wait_until_done(timeout_s=timeout_s)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure_pin(self, pin: int, cfg: TTLTriggerConfig) -> None:
        """Programme a single Digital Out channel (pin)."""
        clock_hz = self._clock_hz

        # Enable the channel
        self._check(
            self._dwf.FDwfDigitalOutEnableSet(self._hdwf, pin, 1),
            f"FDwfDigitalOutEnableSet(pin={pin})",
        )

        # Output type: pulse (hardware counter-based)
        self._check(
            self._dwf.FDwfDigitalOutTypeSet(self._hdwf, pin, DwfDigitalOutTypePulse),
            f"FDwfDigitalOutTypeSet(pin={pin})",
        )

        # Idle state
        self._check(
            self._dwf.FDwfDigitalOutIdleSet(self._hdwf, pin, cfg.idle_state.value),
            f"FDwfDigitalOutIdleSet(pin={pin})",
        )

        # Polarity: ACTIVE_LOW means we invert the pin output
        invert = 1 if cfg.polarity == TTLPolarity.ACTIVE_LOW else 0

        # Convert seconds to clock counts (must be >= 1)
        high_cnt, low_cnt = self._seconds_to_counts(
            cfg.high_time_s, cfg.low_time_s, cfg.delay_s, clock_hz
        )

        # Set the divider to 1 (full clock resolution)
        self._check(
            self._dwf.FDwfDigitalOutDividerSet(self._hdwf, pin, ctypes.c_uint(1)),
            f"FDwfDigitalOutDividerSet(pin={pin})",
        )

        # Set counter: high count, low count
        # FDwfDigitalOutCounterSet(hdwf, channel, low_count, high_count)
        # Note: the DWF API uses (low, high) order — counter starts in low.
        # We swap for ACTIVE_LOW polarity so the initial state matches idle.
        if cfg.polarity == TTLPolarity.ACTIVE_HIGH:
            # Start low (idle), go high for high_cnt, return low for low_cnt
            init_low  = low_cnt
            init_high = high_cnt
        else:
            # ACTIVE_LOW: start high (idle), go low for high_cnt, return high
            init_low  = high_cnt   # this is really the "active" duration
            init_high = low_cnt    # this is the "idle-high" duration

        self._check(
            self._dwf.FDwfDigitalOutCounterSet(
                self._hdwf, pin,
                ctypes.c_uint(init_low),
                ctypes.c_uint(init_high),
            ),
            f"FDwfDigitalOutCounterSet(pin={pin})",
        )

        # Apply pre-trigger delay via the initial-count feature if supported
        if cfg.delay_s > 0.0:
            delay_cnt = max(1, int(round(cfg.delay_s * clock_hz)))
            # FDwfDigitalOutCounterInitSet holds the first half-period count
            # at its initial value for delay_cnt extra clocks
            ret = self._dwf.FDwfDigitalOutCounterInitSet(
                self._hdwf, pin, invert, ctypes.c_uint(delay_cnt)
            )
            if ret == 0:
                # Older firmware / AD2: CounterInitSet may be unsupported.
                # Fall back: increase low count to absorb the delay.
                _log.warning(
                    "FDwfDigitalOutCounterInitSet not supported on this firmware. "
                    "Absorbing delay into low count (timing accuracy reduced)."
                )
                new_low = init_low + delay_cnt
                self._check(
                    self._dwf.FDwfDigitalOutCounterSet(
                        self._hdwf, pin,
                        ctypes.c_uint(new_low),
                        ctypes.c_uint(init_high),
                    ),
                    f"FDwfDigitalOutCounterSet fallback(pin={pin})",
                )

        _log.debug(
            "Pin %d: high_cnt=%d  low_cnt=%d  invert=%d  idle=%s",
            pin, high_cnt, low_cnt, invert, cfg.idle_state.name,
        )

    @staticmethod
    def _seconds_to_counts(
        high_s: float,
        low_s: float,
        delay_s: float,
        clock_hz: float,
    ) -> tuple[int, int]:
        """Convert high/low durations to integer clock counts (min 1 each)."""
        high_cnt = max(1, int(round(high_s * clock_hz)))
        low_cnt  = max(1, int(round(low_s  * clock_hz))) if low_s > 0 else 1
        return high_cnt, low_cnt

    def _read_internal_clock(self) -> float:
        """Query the device's Digital Out internal clock frequency in Hz."""
        hz = ctypes.c_double(0.0)
        ret = self._dwf.FDwfDigitalOutInternalClockInfo(self._hdwf, ctypes.byref(hz))
        if ret != 0 and hz.value > 0:
            return hz.value
        _log.warning(
            "Could not read Digital Out clock; assuming %.0f Hz.",
            self._FALLBACK_CLOCK_HZ,
        )
        return self._FALLBACK_CLOCK_HZ

    def _try_set_voltage(self, voltage_v: float) -> None:
        """
        Attempt to set the Digital Out IO bank voltage.
        Silently ignored on devices that do not support this feature (e.g. AD2).
        """
        ret = self._dwf.FDwfAnalogIOChannelNodeSet(
            self._hdwf,
            ctypes.c_int(2),        # channel 2 = digital voltage on ADP3450
            ctypes.c_int(0),        # node 0 = voltage
            ctypes.c_double(voltage_v),
        )
        if ret != 0:
            _log.debug("Digital IO voltage set to %.2f V.", voltage_v)
        # No error raised if unsupported; AD2 is fixed at 3.3 V.

    def _check(self, ret: int, context: str = "") -> None:
        """Raise DWFError if a DWF call returned 0."""
        if ret == 0:
            msg_buf = ctypes.create_string_buffer(512)
            self._dwf.FDwfGetLastErrorMsg(msg_buf)
            err = msg_buf.value.decode(errors="replace").strip()
            raise DWFError(f"DWF call failed [{context}]: {err}")

    def __repr__(self) -> str:
        if self._cfg is None:
            return "<TTLTrigger unconfigured>"
        return (
            f"<TTLTrigger pins={self._cfg.pins}  "
            f"{self._cfg.high_time_s*1e6:.2f}µs/{self._cfg.low_time_s*1e6:.2f}µs  "
            f"count={self._cfg.pulse_count}  mode={self._cfg.mode.value}>"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== TTL Trigger Demo ===\n")
    print("Connected ADS devices:")
    for d in WaveFormsADS.enumerate():
        print(f"  [{d['index']}] {d['name']}  SN:{d['serial']}  open:{d['is_open']}")

    with WaveFormsADS() as dev:
        trig = TTLTrigger(dev)

        # ---------------------------------------------------------------
        # Example 1: single 50 µs pulse on pin 0
        # ---------------------------------------------------------------
        print("\n[1] Single 50 µs pulse on DIO pin 0 …")
        trig.send_pulse(pin=0, high_time_s=50e-6)
        print("    Done.")

        # ---------------------------------------------------------------
        # Example 2: 1 kHz pulse train, 10% duty cycle, 20 pulses, pins 0+1
        # ---------------------------------------------------------------
        print("\n[2] 1 kHz / 10% duty / 20 pulses on DIO pins 0 and 1 …")
        trig.send_pulse_train(
            pins=[0, 1],
            high_time_s=100e-6,    # 100 µs HIGH
            low_time_s=900e-6,     # 900 µs LOW  → 1 kHz
            pulse_count=20,
            idle_state=TTLIdleState.LOW,
            block=True,
        )
        print("    Done.")

        # ---------------------------------------------------------------
        # Example 3: SOFTWARE-triggered single pulse (arm early, fire fast)
        # ---------------------------------------------------------------
        print("\n[3] Software-triggered 10 µs pulse on pin 0 (pre-armed) …")
        cfg = TTLTriggerConfig(
            pins=[0],
            high_time_s=10e-6,
            low_time_s=0.0,
            pulse_count=1,
            mode=TTLTriggerMode.SOFTWARE,
            delay_s=0.0,
        )
        trig.configure(cfg)
        trig.fire()                        # arms device; no output yet
        print("    Device armed.  Firing software trigger in 0.5 s …")
        time.sleep(0.5)
        trig.fire_software_trigger()       # <~2 ms to first edge
        trig.wait_until_done(timeout_s=1.0)
        print("    Done.")

        # ---------------------------------------------------------------
        # Example 4: active-low pulse with 100 µs pre-trigger delay
        # ---------------------------------------------------------------
        print("\n[4] Active-LOW 20 µs pulse with 100 µs delay on pin 2 …")
        cfg4 = TTLTriggerConfig(
            pins=[2],
            high_time_s=20e-6,
            low_time_s=0.0,
            pulse_count=1,
            idle_state=TTLIdleState.HIGH,
            polarity=TTLPolarity.ACTIVE_LOW,
            delay_s=100e-6,
            amplitude_v=3.3,
        )
        trig.configure(cfg4)
        trig.fire()
        trig.wait_until_done(timeout_s=1.0)
        print("    Done.")

        trig.reset()

    print("\nAll examples complete.")
    sys.exit(0)