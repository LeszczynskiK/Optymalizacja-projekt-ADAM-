import numpy as np
import matplotlib.pyplot as plt

# Booth Function
def booth(x1, x2):
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

# ADAM Optimization Function with Penalty and Constraints
def adam_optimization_with_penalty_and_constraints(gradient_func, function, constraint_func, initial_point, max_learning_rate=0.4, min_learning_rate=0.00001, beta1=0.9, beta2=0.899, epsilon=1e-8, max_iter=500, tol=1e-8, r=25):
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
        
        # Check if the new point violates the constraints
        if constraint_func(*point) < 0:
            print("Optimization terminated because the solution went outside the constraint region.")
            break
        
    return point, np.array(history), np.array(variable_history)

# Gradient Functions
def booth_gradient(x1, x2):
    grad_x1 = 2 * (x1 + 2*x2 - 7) + 2 * (2*x1 + x2 - 5) * 2
    grad_x2 = 2 * (x1 + 2*x2 - 7) * 2 + 2 * (2*x1 + x2 - 5)
    return np.array([grad_x1, grad_x2])

# Constraint Function for Booth
def constraint_g2(x1, x2):
    return x1**2 + 2 * x1 - x2

# Initial point
initial_point = np.array([2, -2.5])

# Run ADAM optimization for Booth Function with Penalty and Constraints
optimal_point_booth, history_booth, variable_history_booth = adam_optimization_with_penalty_and_constraints(booth_gradient, booth, constraint_g2, initial_point)

# Visualize function value changes over iterations - Booth Function
plt.figure(figsize=(10, 6))
plt.plot(history_booth, label='Booth Function')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Booth Function Value vs Iteration')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the contour plot of Booth function with optimal path and constraints
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, booth(X1, X2), levels=np.logspace(-1, 3, 20))
plt.plot(variable_history_booth[:,0], variable_history_booth[:,1], marker='o', color='blue', label='Booth Optimal Path')
plt.plot(initial_point[0], initial_point[1], 'bo')  # Initial point
plt.plot(optimal_point_booth[0], optimal_point_booth[1], 'ro')  # Optimal point - Booth
plt.text(initial_point[0], initial_point[1], 'Initial Point', fontsize=12)
plt.text(optimal_point_booth[0], optimal_point_booth[1], 'Booth Optimal Point', fontsize=12)

# Add constraint g2(x1, x2) <= 0
x1_constraint_booth = np.linspace(-10, 10, 100)
x2_constraint_booth = constraint_g2(x1_constraint_booth, 0)  # Provide the second argument as 0
plt.plot(x1_constraint_booth, x2_constraint_booth, 'g--', label='Constraint g2(x1, x2) <= 0')
plt.fill_between(x1_constraint_booth, x2_constraint_booth, -40, color='skyblue', alpha=0.3)  # Light blue color for the constraint region

# Adjust plot limits for better visualization
plt.xlim(-7.5, 7.5)
plt.ylim(-10, 10)

plt.title('Contour Plot of Booth Function with Penalty')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Check if the optimal point is within the constraint region
if constraint_g2(*optimal_point_booth) >= 0:
    print("Optimal point found within the constraint region.")
else:
    print("Optimal point found outside the constraint region.")

# Print optimization summary
print("Number of iterations:", len(history_booth))
print("Booth Optimal Point:", optimal_point_booth)
print("Function Value at Booth Optimal Point:", booth(*optimal_point_booth))
print("Constraint Value at Booth Optimal Point:", constraint_g2(*optimal_point_booth))
