import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from tabulate import tabulate  # type: ignore

# Define the ODE
def f(x, y):
    #return y/(np.exp(x) - 1)
    return (x*x) + y

# Initial conditions
x0, y0 = 0, 1
h = 0.1
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

# Part 2: RKF45 using solve_ivp for 1000 steps
start_time = time.perf_counter()
sol = solve_ivp(f, [x0, xf], [y0], method='RK45', t_eval=x_eval)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\nRKF45 solved {num_steps} steps in {elapsed_time:.8f} seconds")

# Plot comparison
plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], label="RKF45 Solution")
plt.scatter(x_manual, y_manual, color='red', label='Manual RK4 Points', zorder=5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("ODE: Manual RK4 vs RKF45")
plt.legend()
plt.grid(True)

# Zoom into the initial region to make manual points visible
plt.xlim(x0, x0 + 0.1*num_steps*h)
plt.ylim(min(y_manual)-1, max(y_manual)+1)
plt.show()
