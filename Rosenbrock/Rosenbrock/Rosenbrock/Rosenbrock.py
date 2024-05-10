import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock Function
def rosenbrock(x1, x2, a=1, b=100):
    return (a - x1)**2 + b * (x2 - x1**2)**2

# Adam Optimization with Penalty Function
def adam_optimization(gradient_func, function, constraint_func, initial_point, max_learning_rate=0.9, min_learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100, tol=1e-8, r=25):
    # Initialize parameters
    m = np.zeros_like(initial_point)
    v = np.zeros_like(initial_point)
    t = 0
    point = initial_point
    history = []
    variable_history = [initial_point]

    while t < max_iter:
        t += 1
        gradient = gradient_func(*point)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        learning_rate = max_learning_rate - (max_learning_rate - min_learning_rate) * t / max_iter
        update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Penalty Function for Constraints
        constraint_penalty = constraint_func(*point)
        if constraint_penalty <= 0:
            penalty = 0
        else:
            penalty = r * constraint_penalty
        
        # Add penalty to the function value
        new_point = point - update
        function_value = function(*new_point) + penalty
        
        if np.linalg.norm(update) < tol:
            print("Optimization converged after", t, "iterations.")
            break
        
        point = new_point
        history.append(function_value)  # Save function value
        variable_history.append(point)
        
    return point, np.array(history), np.array(variable_history)

# Gradient Functions
def rosenbrock_gradient(x1, x2, a=1, b=100):
    grad_x1 = -2 * (a - x1) - 4 * b * x1 * (x2 - x1**2)
    grad_x2 = 2 * b * (x2 - x1**2)
    return np.array([grad_x1, grad_x2])

# Constraint Function
def constraint_g1(x1, x2):
    return 1.5 - 0.5 * x1 - x2

# Initial point
initial_point = np.array([-5, 5])

# Run ADAM optimization for Rosenbrock Function with Penalty
optimal_point_rosenbrock, history_rosenbrock, variable_history_rosenbrock = adam_optimization(rosenbrock_gradient, rosenbrock, constraint_g1, initial_point)

# Visualize function value changes over iterations - Rosenbrock Function
plt.figure(figsize=(10, 6))
plt.plot(history_rosenbrock, label='Rosenbrock Function')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Rosenbrock Function Value vs Iteration')
plt.legend()
plt.grid(True)
plt.show()


# Visualize the contour plot of Rosenbrock function with optimal path and constraints
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, rosenbrock(X1, X2), levels=np.logspace(-1, 3, 20))
plt.plot(variable_history_rosenbrock[:,0], variable_history_rosenbrock[:,1], marker='o', color='red', label='Rosenbrock Optimal Path')
plt.plot(initial_point[0], initial_point[1], 'bo')  # Initial point
plt.plot(optimal_point_rosenbrock[0], optimal_point_rosenbrock[1], 'ro')  # Optimal point - Rosenbrock
plt.text(initial_point[0], initial_point[1], 'Initial Point', fontsize=12)
plt.text(optimal_point_rosenbrock[0], optimal_point_rosenbrock[1], 'Rosenbrock Optimal Point', fontsize=12)

# Add constraint g1(x1, x2) = 0
x1_constraint = np.linspace(-10, 10, 100)
x2_constraint = constraint_g1(x1_constraint, 0)  # Provide the second argument as 0
plt.plot(x1_constraint, x2_constraint, 'g--', label='Constraint g1(x1, x2) = 0')
plt.fill_between(x1_constraint, x2_constraint, -10, color='lightblue', alpha=0.3)

plt.title('Contour Plot of Rosenbrock Function with Constraints')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Check if the optimal point is within the constraint region
if constraint_g1(*optimal_point_rosenbrock) >= 0:
    print("Optimal point found within the constraint region.")
else:
    print("Optimal point found outside the constraint region.")

# Print optimization summary
print("Number of iterations:", len(history_rosenbrock))
print("Rosenbrock Optimal Point:", optimal_point_rosenbrock)
print("Function Value at Rosenbrock Optimal Point:", rosenbrock(*optimal_point_rosenbrock))
print("Constraint Value at Rosenbrock Optimal Point:", constraint_g1(*optimal_point_rosenbrock))
