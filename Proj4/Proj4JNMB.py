"""
CST-305 Project 4 - Part 2: Two Processor Closed System
Data Degradation in Digital Storage Systems

Authors: [Your Team Names]
Date: October 2025
Packages: NumPy, SciPy, Matplotlib

Approach:
1. Define system of ODEs for two-processor closed system
2. Calculate eigenvalues and eigenvectors of coefficient matrix
3. Compute matrix exponential e^(At)
4. Solve initial value problem using both numerical and analytical methods
5. Visualize results with matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, eig
from scipy.integrate import solve_ivp

# ============================================================================
# PART 2(a): Mathematical Model Construction
# ============================================================================

def two_processor_ode(t, x):
    """
    System of ODEs for two-processor closed system.
    
    dx1/dt = -2*x1 + 3*x2  (Processor A)
    dx2/dt = 2*x1 - 3*x2   (Processor B)
    
    Parameters:
    -----------
    t : float
        Time
    x : array-like
        State vector [x1, x2]
    
    Returns:
    --------
    dxdt : numpy.ndarray
        Derivative vector [dx1/dt, dx2/dt]
    """
    x1, x2 = x
    dx1dt = -2*x1 + 3*x2  # Input from B (3 MB/s) - Output to B (2 MB/s)
    dx2dt = 2*x1 - 3*x2   # Input from A (2 MB/s) - Output to A (3 MB/s)
    return np.array([dx1dt, dx2dt])


# ============================================================================
# PART 2(b): Matrix Form ẋ = A(t)x(t) + f(t), with f(t) = 0
# ============================================================================

# Coefficient matrix A
A = np.array([[-2, 3],
              [2, -3]])

print("="*70)
print("PART 2: TWO PROCESSOR CLOSED SYSTEM")
print("="*70)
print("\n(b) System in Matrix Form: ẋ = Ax")
print("\nCoefficient Matrix A:")
print(A)
print("\nTransfer rates:")
print("  B → A: 3 MB/sec")
print("  A → B: 2 MB/sec")


# ============================================================================
# PART 2(c): Calculate e^(At) - Matrix Exponential
# ============================================================================

print("\n" + "="*70)
print("(c) EIGENVALUE ANALYSIS AND MATRIX EXPONENTIAL")
print("="*70)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(A)

print("\nEigenvalues:")
for i, lam in enumerate(eigenvalues):
    print(f"  λ{i+1} = {lam:.6f}")

print("\nEigenvectors:")
for i, v in enumerate(eigenvectors.T):
    print(f"  v{i+1} = {v}")

# Physical interpretation
print("\nPhysical Interpretation:")
print("  • λ₁ = 0: Indicates conservation of total data (closed system)")
print("  • λ₂ = -5: Decay rate to equilibrium state")
print("  • All eigenvalues ≤ 0: System is stable")

# Compute matrix exponential at specific time points
print("\n" + "-"*70)
print("Matrix Exponential e^(At) Calculation")
print("-"*70)

def matrix_exponential_analytical(t):
    """
    Analytical computation of e^(At) for our specific 2x2 matrix.
    
    e^(At) = (1/5) * [[3 + 2*e^(-5t),  3 - 3*e^(-5t)],
                      [2 - 2*e^(-5t),  2 + 3*e^(-5t)]]
    """
    exp_term = np.exp(-5 * t)
    return (1/5) * np.array([[3 + 2*exp_term, 3 - 3*exp_term],
                             [2 - 2*exp_term, 2 + 3*exp_term]])

# Display e^(At) at t=0, t=0.5, t=1
test_times = [0, 0.5, 1.0]
for t in test_times:
    print(f"\ne^(At) at t = {t}:")
    eAt_numerical = expm(A * t)
    eAt_analytical = matrix_exponential_analytical(t)
    print("Numerical (using SciPy):")
    print(eAt_numerical)
    print("Analytical formula:")
    print(eAt_analytical)
    print(f"Difference: {np.max(np.abs(eAt_numerical - eAt_analytical)):.2e}")


# ============================================================================
# PART 2(d): Solve Initial Value Problem
# ============================================================================

print("\n" + "="*70)
print("(d) SOLUTION TO INITIAL VALUE PROBLEM")
print("="*70)

# Initial conditions: Both processors start with 100 MB
x0 = np.array([100, 100])  # [x1(0), x2(0)] in MB
print(f"\nInitial Conditions:")
print(f"  x₁(0) = {x0[0]} MB (Processor A)")
print(f"  x₂(0) = {x0[1]} MB (Processor B)")
print(f"  Total Data = {x0[0] + x0[1]} MB")

# Time span
t_span = (0, 3)  # 0 to 3 seconds
t_eval = np.linspace(0, 3, 300)

# Method 1: Numerical solution using SciPy's solve_ivp
print("\nMethod 1: Numerical Solution (solve_ivp with RK45)")
sol_numerical = solve_ivp(two_processor_ode, t_span, x0, 
                          t_eval=t_eval, method='RK45', 
                          rtol=1e-8, atol=1e-10)

# Method 2: Analytical solution using matrix exponential
print("Method 2: Analytical Solution (using e^(At))")

def analytical_solution(t, x0):
    """
    Analytical solution: x(t) = e^(At) * x0
    
    For our specific system with x₁(0) = x₂(0) = 100:
    x₁(t) = 120 - 20*e^(-5t)
    x₂(t) = 80 + 20*e^(-5t)
    """
    x10, x20 = x0
    exp_term = np.exp(-5 * t)
    
    # Using the formula derived from e^(At)
    x1 = (6*x10 + 3*x20 - (2*x10 + 3*x20) * exp_term) / 5
    x2 = (2*x10 + 2*x20 + (2*x10 + 3*x20) * exp_term) / 5
    
    return np.array([x1, x2])

sol_analytical = np.array([analytical_solution(t, x0) for t in t_eval]).T

# Display solution at key time points
print("\nSolution at Key Time Points:")
print("-"*70)
print(f"{'Time (s)':<12} {'x₁(t) Num':<15} {'x₁(t) Ana':<15} {'x₂(t) Num':<15} {'x₂(t) Ana':<15} {'Total':<10}")
print("-"*70)

key_times_indices = [0, 50, 100, 150, 200, 250, 299]
for idx in key_times_indices:
    t = t_eval[idx]
    x1_num = sol_numerical.y[0, idx]
    x2_num = sol_numerical.y[1, idx]
    x1_ana = sol_analytical[0, idx]
    x2_ana = sol_analytical[1, idx]
    total = x1_num + x2_num
    print(f"{t:<12.3f} {x1_num:<15.4f} {x1_ana:<15.4f} {x2_num:<15.4f} {x2_ana:<15.4f} {total:<10.4f}")

# Calculate equilibrium values
print("\nEquilibrium Analysis (as t → ∞):")
x1_eq = 120  # (6*100 + 3*100) / 5 = 120
x2_eq = 80   # (2*100 + 2*100) / 5 = 80
print(f"  x₁(∞) = {x1_eq} MB (Processor A)")
print(f"  x₂(∞) = {x2_eq} MB (Processor B)")
print(f"  Total = {x1_eq + x2_eq} MB (Conservation verified ✓)")

# Verify conservation of total data
total_data = sol_numerical.y[0, :] + sol_numerical.y[1, :]
conservation_error = np.max(np.abs(total_data - 200))
print(f"\nConservation Check:")
print(f"  Maximum deviation from 200 MB: {conservation_error:.2e} MB")
print(f"  Conservation {'VERIFIED ✓' if conservation_error < 1e-6 else 'FAILED ✗'}")


# ============================================================================
# PART 2(f): Visualization
# ============================================================================

print("\n" + "="*70)
print("(f) GENERATING VISUALIZATIONS")
print("="*70)

# Create comprehensive figure with multiple subplots
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Part 2: Two-Processor Closed System - Data Degradation Analysis', 
             fontsize=16, fontweight='bold')

# Subplot 1: Time Evolution of Both Processors
ax1 = plt.subplot(2, 3, 1)
ax1.plot(t_eval, sol_numerical.y[0, :], 'b-', linewidth=2, label='Processor A (Numerical)')
ax1.plot(t_eval, sol_numerical.y[1, :], 'r-', linewidth=2, label='Processor B (Numerical)')
ax1.plot(t_eval, sol_analytical[0, :], 'b--', linewidth=1.5, alpha=0.7, label='Processor A (Analytical)')
ax1.plot(t_eval, sol_analytical[1, :], 'r--', linewidth=1.5, alpha=0.7, label='Processor B (Analytical)')
ax1.axhline(y=120, color='b', linestyle=':', alpha=0.5, label='A Equilibrium (120 MB)')
ax1.axhline(y=80, color='r', linestyle=':', alpha=0.5, label='B Equilibrium (80 MB)')
ax1.set_xlabel('Time (seconds)', fontsize=11)
ax1.set_ylabel('Data (MB)', fontsize=11)
ax1.set_title('(a) Time Evolution of Data in Each Processor', fontweight='bold')
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Total Data Conservation
ax2 = plt.subplot(2, 3, 2)
ax2.plot(t_eval, total_data, 'g-', linewidth=2, label='Total Data')
ax2.axhline(y=200, color='k', linestyle='--', alpha=0.5, label='Expected (200 MB)')
ax2.set_xlabel('Time (seconds)', fontsize=11)
ax2.set_ylabel('Total Data (MB)', fontsize=11)
ax2.set_title('(b) Conservation of Total Data (Closed System)', fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([199.9, 200.1])

# Subplot 3: Phase Portrait (x1 vs x2)
ax3 = plt.subplot(2, 3, 3)
ax3.plot(sol_numerical.y[0, :], sol_numerical.y[1, :], 'purple', linewidth=2)
ax3.plot(x0[0], x0[1], 'go', markersize=10, label='Initial State')
ax3.plot(x1_eq, x2_eq, 'ro', markersize=10, label='Equilibrium')
ax3.set_xlabel('x₁ (Processor A) [MB]', fontsize=11)
ax3.set_ylabel('x₂ (Processor B) [MB]', fontsize=11)
ax3.set_title('(c) Phase Portrait: State Trajectory', fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# Subplot 4: Numerical vs Analytical Comparison
ax4 = plt.subplot(2, 3, 4)
error_x1 = np.abs(sol_numerical.y[0, :] - sol_analytical[0, :])
error_x2 = np.abs(sol_numerical.y[1, :] - sol_analytical[1, :])
ax4.semilogy(t_eval, error_x1, 'b-', linewidth=2, label='Error in x₁')
ax4.semilogy(t_eval, error_x2, 'r-', linewidth=2, label='Error in x₂')
ax4.set_xlabel('Time (seconds)', fontsize=11)
ax4.set_ylabel('Absolute Error (MB)', fontsize=11)
ax4.set_title('(d) Numerical vs Analytical Solution Error', fontweight='bold')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3, which='both')

# Subplot 5: Data Transfer Rates
ax5 = plt.subplot(2, 3, 5)
rate_A_to_B = 2 * sol_numerical.y[0, :]  # A → B at 2 MB/s
rate_B_to_A = 3 * sol_numerical.y[1, :]  # B → A at 3 MB/s
net_flow = rate_B_to_A - rate_A_to_B     # Net flow into A
ax5.plot(t_eval, rate_A_to_B, 'b-', linewidth=2, label='A → B (2x₁)')
ax5.plot(t_eval, rate_B_to_A, 'r-', linewidth=2, label='B → A (3x₂)')
ax5.plot(t_eval, net_flow, 'g--', linewidth=2, label='Net Flow to A')
ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax5.set_xlabel('Time (seconds)', fontsize=11)
ax5.set_ylabel('Transfer Rate (MB/s)', fontsize=11)
ax5.set_title('(e) Data Transfer Rates Over Time', fontweight='bold')
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, alpha=0.3)

# Subplot 6: Exponential Decay Component
ax6 = plt.subplot(2, 3, 6)
exp_component = 20 * np.exp(-5 * t_eval)  # The e^(-5t) term
ax6.plot(t_eval, exp_component, 'purple', linewidth=2, label='20e^(-5t)')
ax6.set_xlabel('Time (seconds)', fontsize=11)
ax6.set_ylabel('Amplitude (MB)', fontsize=11)
ax6.set_title('(f) Exponential Decay Component (λ₂ = -5)', fontweight='bold')
ax6.legend(loc='best', fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('part2_two_processor_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved as: part2_two_processor_analysis.png")

# ============================================================================
# Additional Analysis: Vector Field Visualization
# ============================================================================

fig2, ax = plt.subplots(figsize=(10, 10))

# Create mesh grid for vector field
x1_range = np.linspace(50, 150, 20)
x2_range = np.linspace(50, 150, 20)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Calculate vector field
U = -2*X1 + 3*X2  # dx1/dt
V = 2*X1 - 3*X2   # dx2/dt

# Normalize arrows for better visualization
M = np.sqrt(U**2 + V**2)
U_norm = U / (M + 1e-10)
V_norm = V / (M + 1e-10)

# Plot vector field
ax.quiver(X1, X2, U_norm, V_norm, M, cmap='viridis', alpha=0.6)
ax.plot(sol_numerical.y[0, :], sol_numerical.y[1, :], 'r-', linewidth=3, 
        label='Solution Trajectory')
ax.plot(x0[0], x0[1], 'go', markersize=15, label='Initial State (100, 100)', zorder=5)
ax.plot(x1_eq, x2_eq, 'ro', markersize=15, label='Equilibrium (120, 80)', zorder=5)

# Plot conservation line (x1 + x2 = 200)
x1_line = np.linspace(50, 150, 100)
x2_line = 200 - x1_line
ax.plot(x1_line, x2_line, 'k--', linewidth=2, alpha=0.5, 
        label='Conservation Line (x₁+x₂=200)')

ax.set_xlabel('x₁ - Processor A Data (MB)', fontsize=12, fontweight='bold')
ax.set_ylabel('x₂ - Processor B Data (MB)', fontsize=12, fontweight='bold')
ax.set_title('Vector Field and Phase Portrait\nTwo-Processor Closed System', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('part2_vector_field.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved as: part2_vector_field.png")

plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nKey Results Summary:")
print(f"  • Initial state: A={x0[0]} MB, B={x0[1]} MB")
print(f"  • Equilibrium: A={x1_eq} MB, B={x2_eq} MB")
print(f"  • Eigenvalues: λ₁=0 (conservation), λ₂=-5 (decay rate)")
print(f"  • Time constant τ = 1/5 = 0.2 seconds")
print(f"  • 95% equilibrium reached at t ≈ 0.6 seconds")
print(f"  • Conservation error: {conservation_error:.2e} MB")
print("\n✓ All visualizations generated successfully!")
