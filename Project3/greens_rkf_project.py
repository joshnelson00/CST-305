#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CST-305 Project 3 — Green's Function + RKF (RK45) Implementation
Team: <add names> | Responsibilities: <brief roles>

System Context: Compare analytic (Green’s function) vs RKF numerical performance on two
2nd-order LTI ODEs with zero ICs, t ≥ 0.

Assigned IVPs:
1) y'' + 2y' + y = 2t,  y(0)=0, y'(0)=0
2) y'' + y       = t^2,  y(0)=0, y'(0)=0

Deliverables covered:
- Formal solutions (via Green’s function; verified vs Undetermined Coefficients / Var. of Params)
- RK45 numerical solve + plots (homogeneous fundamentals + non-homogeneous solutions)
- Clean CLI and comments

Run:
    python3 greens_rkf_project.py --tmax 10 --n 2000   # high-res
    python3 greens_rkf_project.py --ode both           # 1, 2, or both
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------
# Problem setup
# ----------------------------
def rhs1(t):
    """f(t) for ODE1: y'' + 2y' + y = 2t"""
    return 2.0 * t

def rhs2(t):
    """f(t) for ODE2: y'' + y = t^2"""
    return t**2

def greens1(t, tau):
    """Green's kernel for ODE1: critically damped system"""
    dt = t - tau
    return np.where(dt >= 0.0, dt * np.exp(-dt), 0.0)

def greens2(t, tau):
    """Green's kernel for ODE2: undamped oscillator"""
    dt = t - tau
    return np.where(dt >= 0.0, np.sin(dt), 0.0)

# Closed-form analytic solutions (verified manually or with symbolic tools)
def closed_form_ode1(t):
    # y = (4 + 2t) e^{-t} + 2t - 4
    return (4.0 + 2.0*t)*np.exp(-t) + 2.0*t - 4.0

def closed_form_ode2(t):
    # y = 2 cos t + t^2 - 2
    return 2.0*np.cos(t) + t**2 - 2.0

# Homogeneous (fundamental) solutions for plotting
def homo1_fundamental(t):
    # y'' + 2y' + y = 0  → roots r=-1 (double)
    phi1 = (1.0 - t)*np.exp(-t)
    phi2 = t*np.exp(-t)
    return phi1, phi2

def homo2_fundamental(t):
    # y'' + y = 0
    phi1 = np.cos(t)
    phi2 = np.sin(t)
    return phi1, phi2

# ----------------------------
# Green’s Function Convolution
# ----------------------------
def greens_convolution(t_grid, f_vals, kernel):
    """
    Compute y(t_i) = ∫_0^{t_i} G(t_i, τ) f(τ) dτ
    using trapezoidal rule on a fixed grid.
    """
    t = t_grid
    n = t.size
    y = np.zeros(n)
    dt = np.diff(t)
    for i in range(1, n):
        tau = t[:i+1]
        k = kernel(t[i], tau)
        f = f_vals[:i+1]
        kj, fj = k[:-1], f[:-1]
        kj1, fj1 = k[1:], f[1:]
        y[i] = np.sum(0.5*(kj*fj + kj1*fj1)*dt[:i])
    return y

# ----------------------------
# RK45 (RKF) Numerical Solvers
# ----------------------------
def rkf_ode1(t_span, t_eval):
    def sys(t, Y):
        y, v = Y
        return [v, -2.0*v - y + 2.0*t]
    sol = solve_ivp(sys, t_span, [0.0, 0.0], t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10)
    return sol.y[0]

def rkf_ode2(t_span, t_eval):
    def sys(t, Y):
        y, v = Y
        return [v, -y + t**2]
    sol = solve_ivp(sys, t_span, [0.0, 0.0], t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10)
    return sol.y[0]

# ----------------------------
# Plotting
# ----------------------------
def plot_all(t, ode_id, yh_fund, y_green, y_closed, y_rkf):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Left: homogeneous fundamentals
    phi1, phi2 = yh_fund(t)
    ax[0].plot(t, phi1, label=r'$\phi_1$ (homogeneous)')
    ax[0].plot(t, phi2, '--', label=r'$\phi_2$ (homogeneous)')
    ax[0].set_title(f'ODE {ode_id}: Homogeneous Fundamentals')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('Response')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Right: solution comparison
    ax[1].plot(t, y_green, label="Green's function")
    ax[1].plot(t, y_closed, '--', label='Closed-form')
    ax[1].plot(t, y_rkf, label='RK45 (RKF)', linewidth=1)
    ax[1].set_title(f'ODE {ode_id}: Solution Comparison')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('y(t)')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Green's Function vs RKF on two IVPs")
    parser.add_argument('--tmax', type=float, default=10.0, help='max time')
    parser.add_argument('--n', type=int, default=1000, help='grid points')
    parser.add_argument('--ode', type=str, default='both', choices=['1', '2', 'both'], help='which ODE to run')
    args = parser.parse_args()

    t = np.linspace(0.0, args.tmax, args.n)

    # Print reference closed-form solutions
    print("[Analytic] ODE1 closed form: y = (4 + 2t)e^{-t} + 2t - 4")
    print("[Analytic] ODE2 closed form: y = 2cos(t) + t^2 - 2")

    if args.ode in ('1', 'both'):
        f1 = rhs1(t)
        y1_green = greens_convolution(t, f1, greens1)
        y1_closed = closed_form_ode1(t)
        y1_rkf = rkf_ode1((t[0], t[-1]), t)

        err1 = np.max(np.abs(y1_green - y1_closed))
        err1_num = np.max(np.abs(y1_rkf - y1_closed))
        print(f"[ODE1] max|Green - Closed| = {err1:.3e}, max|RKF - Closed| = {err1_num:.3e}")

        plot_all(t, ode_id=1, yh_fund=homo1_fundamental,
                 y_green=y1_green, y_closed=y1_closed, y_rkf=y1_rkf)

    if args.ode in ('2', 'both'):
        f2 = rhs2(t)
        y2_green = greens_convolution(t, f2, greens2)
        y2_closed = closed_form_ode2(t)
        y2_rkf = rkf_ode2((t[0], t[-1]), t)

        err2 = np.max(np.abs(y2_green - y2_closed))
        err2_num = np.max(np.abs(y2_rkf - y2_closed))
        print(f"[ODE2] max|Green - Closed| = {err2:.3e}, max|RKF - Closed| = {err2_num:.3e}")

        plot_all(t, ode_id=2, yh_fund=homo2_fundamental,
                 y_green=y2_green, y_closed=y2_closed, y_rkf=y2_rkf)

if __name__ == "__main__":
    main()
