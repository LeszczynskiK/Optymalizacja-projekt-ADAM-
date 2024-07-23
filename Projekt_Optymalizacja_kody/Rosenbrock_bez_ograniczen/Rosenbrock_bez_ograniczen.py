import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock Function
def rosenbrock(x, a=1, b=100):
    x1, x2 = x
    return (a - x1)**2 + b * (x2 - x1**2)**2

# Gradient Functions for Rosenbrock
def rosenbrock_gradient(x, a=1, b=100):
    x1, x2 = x
    grad_x1 = -2 * (a - x1) - 4 * b * x1 * (x2 - x1**2)
    grad_x2 = 2 * b * (x2 - x1**2)
    return np.array([grad_x1, grad_x2])

# ADAM Optimization Function with max and min learning rates
def adam_optimization(gradient_func, function, initial_point, max_learning_rate=0.999, min_learning_rate=0.001, beta1=0.89, beta2=0.895, epsilon=1e-8, max_iter=250, tol=1e-8):
    m = np.zeros_like(initial_point)
    v = np.zeros_like(initial_point)
    t = 0
    point = initial_point
    history = []
    variable_history = [initial_point]

    while t < max_iter:
        t += 1
        gradient = gradient_func(point)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Linearly decaying learning rate
        learning_rate = max_learning_rate - (max_learning_rate - min_learning_rate) * (t / max_iter)
        
        update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        new_point = point - update
        function_value = function(new_point)
        
        if np.linalg.norm(update) < tol:
            print("Optimization converged after", t, "iterations.")
            break
        
        point = new_point
        history.append(function_value)  # Save function value
        variable_history.append(point)
        
    return point, np.array(history), np.array(variable_history)

# Initial point for Rosenbrock
initial_point_rosenbrock = np.array([-5, 10])

# Run ADAM optimization for Rosenbrock Function
optimal_point_rosenbrock, history_rosenbrock, variable_history_rosenbrock = adam_optimization(rosenbrock_gradient, rosenbrock, initial_point_rosenbrock)

# Visualize function value changes over iterations - Rosenbrock Function
plt.figure(figsize=(10, 6))
plt.plot(history_rosenbrock, label='Rosenbrock Function')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Rosenbrock Function Value vs Iteration')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the contour plot of Rosenbrock function with optimal path
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, rosenbrock([X1, X2]), levels=np.logspace(-1, 3, 20))
plt.plot(variable_history_rosenbrock[:, 0], variable_history_rosenbrock[:, 1], marker='o', color='red', label='Rosenbrock Optimal Path')
plt.plot(initial_point_rosenbrock[0], initial_point_rosenbrock[1], 'bo')  # Initial point
plt.plot(optimal_point_rosenbrock[0], optimal_point_rosenbrock[1], 'ro')  # Optimal point - Rosenbrock
plt.text(initial_point_rosenbrock[0], initial_point_rosenbrock[1], 'Initial Point', fontsize=12)
plt.text(optimal_point_rosenbrock[0], optimal_point_rosenbrock[1], 'Rosenbrock Optimal Point', fontsize=12)

plt.title('Contour Plot of Rosenbrock Function')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Print optimization summary
print("Number of iterations:", len(history_rosenbrock))
print("Rosenbrock Optimal Point:", optimal_point_rosenbrock)
print("Function Value at Rosenbrock Optimal Point:", rosenbrock(optimal_point_rosenbrock))
