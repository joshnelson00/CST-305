import matplotlib
import matplotlib.pyplot as plt 
from scipy.integrate import odeint 
import numpy as np 


"""
Course: CST-305: Computational Modeling and Simulation
Project: Visualizing an ODE for Battery Charging
Author(s): Josh Nelson, Martin Battu, Karik Freiheit

Purpose: Create and Model an ODE with Python that relates to a function in a computer.

Packages: matplotlib, scipy, numpy
"""

# ODE for battery charging
# dQ/dt = (Qmax - Q) / (Resistance*Capacitance (AKA RC))
def dQ_dt(t, Q, Qmax=100, RC=5):
    return (Qmax - Q) / RC

# Initial charge (battery starts at 0%)
Q0 = 0

# Time range (e.g., 0 to 50 units)
t = np.linspace(0, 50, 200)

# Solve ODE
Q = odeint(dQ_dt, Q0, t, tfirst=True)

# Plot solution
plt.plot(t, Q, label="Battery Charge")
plt.xlabel('Time', fontsize=22)
plt.ylabel('Charge (%)', fontsize=22)
plt.title("Battery Charging Curve", fontsize=18)
plt.legend()
plt.show()
