import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Problem Setup: Minimize f(x, y, z) = (x - 3)^2 + (y + 1)^2 + (z - 2)^2, subject to x, y, z >= 0

# Parameters
b = np.array([3, -1, 2])  # Example target vector

# Objective Function (Quadratic loss)
def objective(x, y, z):
    return (x - 3)**2 + (y + 1)**2 + (z - 2)**2

# Gradient of the Objective Function with respect to x, y, z
def gradient(x, y, z):
    return np.array([
        2 * (x - 3),
        2 * (y + 1),
        2 * (z - 2)
    ])

# Constraints (x, y, z >= 0, x + y >= 1, and x - y + z = 2)
def projection_feasible_region(x, y, z):
    x = max(0, x)
    y = max(0, y)
    z = max(0, z)
    # Additional inequality constraint: x + y >= 1
    if x + y < 1:
        y = 1 - x if x < 1 else y
    # Equality constraint: x - y + z = 2
    correction = (x - y + z) - 2
    z -= correction / 3
    x -= correction / 3
    y += correction / 3
    return x, y, z

# Frank-Wolfe (FW) Algorithm
def frank_wolfe(x_init, max_iter=100, tol=1e-6):
    x = x_init
    trajectory = [np.array(x)]
    for i in range(max_iter):
        # Linearize the objective
        grad = gradient(*x)
        s = np.zeros_like(x)
        idx = np.argmin(grad)
        s[idx] = 4 if grad[idx] < 0 else 0  # Sparse solution with L1 norm constraint
        direction = s - x
        step_size = 2 / (i + 2)  # Step size rule for FW
        x = x + step_size * direction
        trajectory.append(np.array(x))
        if np.linalg.norm(grad) < tol:
            break
    return x, trajectory

# Projected Gradient Descent
def projected_gradient_descent(x_init, max_iter=100, tol=1e-6, alpha=0.1):
    x = x_init
    trajectory = [np.array(x)]
    for i in range(max_iter):
        grad = gradient(*x)
        x = x - alpha * grad
        # Projection onto feasible region
        x = projection_feasible_region(*x)
        trajectory.append(np.array(x))
        if np.linalg.norm(grad) < tol:
            break
    return x, trajectory

# Initial Point
x_init = (0.5, 0.5, 0.5)

# Run Algorithms
fw_solution, fw_trajectory = frank_wolfe(x_init)
pgd_solution, pgd_trajectory = projected_gradient_descent(x_init)

# Plot Trajectories
plt.figure(figsize=(10, 6))

# Frank-Wolfe Trajectory
fw_trajectory = np.array(fw_trajectory)
plt.plot(range(len(fw_trajectory)), [objective(*x) for x in fw_trajectory], 'o-', label='Frank-Wolfe', color='blue', alpha=0.5, linewidth=1)

# Projected Gradient Descent Trajectory
pgd_trajectory = np.array(pgd_trajectory)
plt.plot(range(len(pgd_trajectory)), [objective(*x) for x in pgd_trajectory], 'd-', label='Projected Gradient Descent', color='red', alpha=0.5, linewidth=2, markersize=4)


# Plot Formatting
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Objective Function Value vs Iterations for Different Optimization Algorithms')
plt.legend()
plt.grid(True)
plt.savefig('../out/frank_wolfe.png')
plt.show()