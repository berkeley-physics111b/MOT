from waveforms_ads import *
import time

with WaveFormsADS() as dev:
    print(f"\nOpened: {dev}")
    print("Testing multiple pulses on one pin")
    dev.digital_out_pulse(0, 0.5, 0.5, pulse_count=2, wait_for_done=True)
    print("Testing pulse train on several pins with different high/low times")
    dev.digital_out_pulse_train([0, 1], [0.5, 0.25], [0.5, 0.75], pulse_count=1, wait_for_done=True)
    print("Testing pulse train on several pins with multiple pulses")
    dev.digital_out_pulse_train([0, 1], [0.5, 0.5], [0.5, 0.5], pulse_count=2, wait_for_done=True)
    print("Pulse train, multiple pulses, dif times")
    dev.digital_out_pulse_train([0, 1], [0.5, 0.25], [0.5, 0.75], pulse_count=2, wait_for_done=True)