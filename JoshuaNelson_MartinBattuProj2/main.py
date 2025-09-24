from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# Parameters for logistic equation
r = 0.5   # growth rate
K = 10    # carrying capacity

# Define ODE system
def rhs(t, y):
    return r * y * (1 - y / K)

# Initial conditions
t0, tf = 0, 20   # time span
y0 = [1]         # initial population

# Solve using RK45
sol = solve_ivp(rhs, [t0, tf], y0, method='RK45', dense_output=True)

# Create time points for plotting
t = np.linspace(t0, tf, 300)
y = sol.sol(t)[0]

# Plot solution
plt.plot(t, y, label="Population y(t)")
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title("Logistic Growth Model")
plt.legend()
plt.grid(True)
plt.show()
