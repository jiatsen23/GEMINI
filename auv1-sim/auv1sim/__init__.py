"""auv1sim - a small physics simulator for AUV1 in a pool.

Package layout (read them in this order):
    config.py     load pool.json / vehicle.json
    dynamics.py   the vehicle: forces -> motion (6 degrees of freedom)
    sensors.py    turn the TRUE state into noisy sensor readings
    estimator.py  turn noisy readings back into an ESTIMATED state
    sim_link.py   the SimLink class: looks like MavlinkIO to the control code
"""
