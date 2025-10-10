# CST-305 Project 3: Green's Function + RKF Implementation

## Overview

This project implements and compares two methods for solving second-order linear ordinary differential equations (ODEs):
1. **Analytical solution** using Green's Function with convolution
2. **Numerical solution** using RKF (Runge-Kutta-Fehlberg) method (RK45)

### Problems Solved

**ODE 1 (Critically Damped System):**
```
y'' + 2y' + y = 2t
y(0) = 0, y'(0) = 0
```

**ODE 2 (Simple Harmonic Oscillator):**
```
y'' + y = t²
y(0) = 0, y'(0) = 0
```

The program:
- Computes solutions using Green's function convolution
- Verifies against closed-form solutions (Undetermined Coefficients/Variation of Parameters)
- Numerically solves using SciPy's RK45 implementation
- Plots homogeneous fundamental solutions and non-homogeneous solution comparisons
- Reports error metrics between methods

## Requirements

### Dependencies
- Python 3.6 or higher
- NumPy
- Matplotlib
- SciPy
- SymPy

## Installation

### Windows

1. **Install Python** (if not already installed):
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Install dependencies**:
   ```bash
   pip install numpy matplotlib scipy sympy
   ```

3. **Verify installation**:
   ```bash
   python --version
   pip list
   ```

### macOS

1. **Install Python** (if not already installed):
   ```bash
   # Using Homebrew (recommended)
   brew install python

   # Or download from python.org
   ```

2. **Install dependencies**:
   ```bash
   pip3 install numpy matplotlib scipy sympy
   ```

3. **Verify installation**:
   ```bash
   python3 --version
   pip3 list
   ```

## Running the Program

### Basic Usage

**Windows:**
```bash
python greens_rkf_project.py
```

**macOS/Linux:**
```bash
python3 greens_rkf_project.py
```

### Command-Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--tmax` | Maximum time for simulation | 10.0 | `--tmax 15` |
| `--n` | Number of grid points | 1000 | `--n 2000` |
| `--ode` | Which ODE to solve: `1`, `2`, or `both` | `both` | `--ode 1` |

### Example Commands

**High-resolution simulation (both ODEs):**
```bash
# Windows
python greens_rkf_project.py --tmax 10 --n 2000

# macOS/Linux
python3 greens_rkf_project.py --tmax 10 --n 2000
```

**Solve only ODE 1:**
```bash
# Windows
python greens_rkf_project.py --ode 1

# macOS/Linux
python3 greens_rkf_project.py --ode 1
```

**Extended time range with fine grid:**
```bash
# Windows
python greens_rkf_project.py --tmax 20 --n 5000 --ode both

# macOS/Linux
python3 greens_rkf_project.py --tmax 20 --n 5000 --ode both
```

## Output

### Console Output
The program displays:
- Symbolic closed-form solutions (via SymPy)
- Maximum error between Green's function and analytical solution
- Maximum error between RKF numerical solution and analytical solution

Example:
```
[SymPy] ODE1 closed form: 2*t*exp(-t) + 2*t + 4*exp(-t) - 4
[SymPy] ODE2 closed form: t**2 + 2*cos(t) - 2
[ODE1] max|Green - Closed| = 1.234e-05, max|RKF - Closed| = 2.345e-09
[ODE2] max|Green - Closed| = 3.456e-06, max|RKF - Closed| = 1.234e-09
```

### Graphical Output
For each ODE, the program generates a figure with two subplots:
- **Left panel**: Homogeneous fundamental solutions (φ₁, φ₂)
- **Right panel**: Comparison of Green's function, closed-form, and RKF solutions

## Troubleshooting

### Common Issues

**"python: command not found" (macOS/Linux)**
- Use `python3` instead of `python`

**"pip: command not found"**
- Windows: Ensure Python was added to PATH during installation
- macOS: Use `pip3` instead of `pip`

**ModuleNotFoundError**
```bash
# Reinstall missing package
pip install <package_name>

# Or reinstall all at once
pip install numpy matplotlib scipy sympy
```

**Matplotlib backend issues (Linux/WSL)**
```bash
# If plots don't appear, try setting backend
export MPLBACKEND=TkAgg
python3 greens_rkf_project.py
```

### Performance Notes
- For `n > 5000`, the Green's function convolution may become slow (O(n²) complexity)
- RKF solver is adaptive and remains efficient for most grid sizes
- Recommended range: `n = 1000` to `5000`

## Project Structure

```
CST-305/
├── greens_rkf_project.py    # Main program
└── README.md                 # This file
```



