import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
import time
from tabulate import tabulate  # type: ignore

# Define the ODE
def f(x, y):
    return y / (np.exp(x) - 1)

# Wrapper for odeint (swaps x and y arguments)
def f_odeint(y, x):
    return y / (np.exp(x) - 1)

# Initial conditions
x0, y0 = 1, 5
h = 0.02
num_steps = 1000
xf = x0 + h*num_steps
x_eval = np.linspace(x0, xf, num_steps)

# Part 1: Manual RK4 for first 6 points with table output
def rk4_manual_table(f, x0, y0, h, steps=5):
    x_values = [x0]
    y_values = [y0]
    table_data = []

    for i in range(steps):
        x_n = x_values[-1]
        y_n = y_values[-1]

        k1 = f(x_n, y_n)
        k2 = f(x_n + h/2, y_n + h/2*k1)
        k3 = f(x_n + h/2, y_n + h/2*k2)
        k4 = f(x_n + h, y_n + h*k3)

        y_next = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x_next = x_n + h

        x_values.append(round(x_next, 5))
        y_values.append(round(y_next, 5))

        # Append all info to table
        table_data.append([i+1, x_n, y_n, k1, k2, k3, k4, y_next])

    # Print table
    headers = ["Step", "x_n", "y_n", "k1", "k2", "k3", "k4", "y_next"]
    print(tabulate(table_data, headers=headers, floatfmt=".6f"))

    return x_values, y_values

x_manual, y_manual = rk4_manual_table(f, x0, y0, h, steps=6)

# Part 2: RK45 using solve_ivp
start_time = time.perf_counter()
sol_rk45 = solve_ivp(f, [x0, xf], [y0], method='RK45', t_eval=x_eval)
end_time = time.perf_counter()
elapsed_time_rk45 = end_time - start_time
print(f"\nRKF45 solved {num_steps} steps in {elapsed_time_rk45:.8f} seconds")

# Part 3: ODEINT solution
start_time = time.perf_counter()
sol_odeint = odeint(f_odeint, y0, x_eval)
end_time = time.perf_counter()
elapsed_time_odeint = end_time - start_time
print(f"ODEINT solved {num_steps} steps in {elapsed_time_odeint:.8f} seconds")

# Plot 1: RK45 only
plt.figure(figsize=(10,6))
plt.plot(sol_rk45.t, sol_rk45.y[0], label="RK45 Solution", color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("ODE Solution using RK45")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: ODEINT only
plt.figure(figsize=(10,6))
plt.plot(x_eval, sol_odeint[:,0], label="ODEINT Solution", color='green')
plt.xlabel("x")
plt.ylabel("y")
plt.title("ODE Solution using ODEINT")
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Overlay RK45 and ODEINT
plt.figure(figsize=(10,6))
plt.plot(sol_rk45.t, sol_rk45.y[0], label="RK45", color='blue')
plt.plot(x_eval, sol_odeint[:,0], label="ODEINT", color='green', linestyle='--')
plt.scatter(x_manual, y_manual, color='red', label='Manual RK4 Points', zorder=5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("ODE Solution: RK45 vs ODEINT vs Manual RK4 Points")
plt.legend()
plt.grid(True)
plt.show()
