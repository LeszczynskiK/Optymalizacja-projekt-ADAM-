import numpy as np
import matplotlib.pyplot as plt

# Three-Hump Camel Function
def three_hump_camel(x1, x2):
    return 2*x1**2 - 1.05*x1**4 + (1/6)*x1**6 + x1*x2 + x2**2

# Gradient Function for Three-Hump Camel Function
def three_hump_camel_gradient(x1, x2):
    grad_x1 = 4 * x1 - 4.2 * x1**3 + x1**5 + x2
    grad_x2 = x1 + 2 * x2
    return np.array([grad_x1, grad_x2])

# Constraint Function for Three-Hump Camel Function
def constraint_g3(x1, x2):
    return x1**2 + x2**2 - 1  # Circle constraint: x1^2 + x2^2 <= 1

# Projection onto the feasible set defined by the constraint
def project_onto_feasible_set(point):
    if constraint_g3(*point) > 0:
        return point / np.linalg.norm(point)  # Project onto the unit circle
    return point

# ADAM Optimization Function with Constraints
def adam_optimization_with_constraints(gradient_func, function, constraint_func, initial_point, max_learning_rate=0.999, min_learning_rate=0.0001, beta1=0.89, beta2=0.895, epsilon=1e-8, max_iter=500, tol=1e-8):
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
        
        new_point = point - update
        new_point = project_onto_feasible_set(new_point)
        
        function_value = function(*new_point)
        
        point = new_point
        history.append(function_value)  # Save function value
        variable_history.append(point)
        
        if np.linalg.norm(update) < tol:
            break
        
    return point, np.array(history), np.array(variable_history)

# Initial point
initial_point = np.array([-5, 5])

# Run ADAM optimization for Three-Hump Camel Function with Constraints
optimal_point_camel, history_camel, variable_history_camel = adam_optimization_with_constraints(three_hump_camel_gradient, three_hump_camel, constraint_g3, initial_point)

# Visualize function value changes over iterations - Three-Hump Camel Function
plt.figure(figsize=(10, 6))
plt.plot(history_camel, label='Three-Hump Camel Function')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Three-Hump Camel Function Value vs Iteration')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the contour plot of Three-Hump Camel function with optimal path and constraints
x1 = np.linspace(-5.5, 5.5, 100)
x2 = np.linspace(-5.5, 5.5, 100)
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, three_hump_camel(X1, X2), levels=np.linspace(-1, 10, 100))
plt.plot(variable_history_camel[:,0], variable_history_camel[:,1], marker='o', color='green', label='Three-Hump Camel Optimal Path')
plt.plot(initial_point[0], initial_point[1], 'bo')  # Initial point
plt.plot(optimal_point_camel[0], optimal_point_camel[1], 'ro')  # Optimal point - Three-Hump Camel
plt.text(initial_point[0], initial_point[1], 'Initial Point', fontsize=12)
plt.text(optimal_point_camel[0], optimal_point_camel[1], 'Three-Hump Camel Optimal Point', fontsize=12)

# Add constraint g3(x1, x2) <= 1
circle_constraint = plt.Circle((0, 0), np.sqrt(1), color='black', alpha=0.7)  # Light blue color for the constraint region
plt.gca().add_patch(circle_constraint)
plt.xlim(-5.5, 5.5)
plt.ylim(-5.5, 5.5)

plt.title('Contour Plot of Three-Hump Camel Function with Constraints')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Check if the optimal point is within the constraint region
if constraint_g3(*optimal_point_camel) <= 0:
    print("Optimal point found within the constraint region.")
else:
    print("Optimal point found outside the constraint region.")

# Print optimization summary
print("Number of iterations:", len(history_camel))
print("Three-Hump Camel Optimal Point:", optimal_point_camel)
print("Function Value at Three-Hump Camel Optimal Point:", three_hump_camel(*optimal_point_camel))
print("Constraint Value at Three-Hump Camel Optimal Point:", constraint_g3(*optimal_point_camel))
