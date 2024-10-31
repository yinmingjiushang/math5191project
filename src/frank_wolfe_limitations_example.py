import numpy as np
import matplotlib.pyplot as plt

# Problem Setup: Minimize f(x, y) = x + (1 - x * y)^2, subject to 0 <= x <= 1, 0 <= y <= 1, x + y >= 1

# Objective Function (non-convex with quadratic term)
def objective(x, y):
    return x + (1 - x * y)**2

# Gradient of the Objective Function with respect to x, y
def gradient(x, y):
    return np.array([
        1 - 2 * y * (1 - x * y),  # derivative with respect to x
        -2 * x * (1 - x * y)      # derivative with respect to y
    ])

# Constraints (x, y >= 0, x + y >= 1)
def projection_feasible_region(x, y):
    # Apply non-negativity constraints
    x = max(0, min(x, 1))  # Ensure x is within [0, 1]
    y = max(0, min(y, 1))  # Ensure y is within [0, 1]
    # Apply additional inequality constraint: x + y >= 1
    if x + y < 1:
        if x < 1:
            y = 1 - x
        else:
            x = 1 - y
    return x, y

# Frank-Wolfe (FW) Algorithm
def frank_wolfe(x_init, max_iter=100, tol=1e-6):
    x = x_init
    trajectory = [np.array(x)]
    for i in range(max_iter):
        # Linearize the objective
        grad = gradient(*x)
        s = np.zeros_like(x)
        idx = np.argmin(grad)
        s[idx] = 1 if grad[idx] < 0 else 0  # Sparse solution for FW
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
x_init = (0.5, 0.5)

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
plt.title('Objective Function Value vs Iterations for f(x, y) = x + (1 - xy)^2')

plt.legend()
plt.grid(True)
plt.savefig('../out/frank_wolfe_limitation.png')
plt.show()
