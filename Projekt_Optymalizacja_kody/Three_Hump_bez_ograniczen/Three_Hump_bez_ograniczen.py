import numpy as np
import matplotlib.pyplot as plt

# Three-Hump Camel Function
def three_hump_camel(x):
    x1, x2 = x
    return 2 * x1**2 - 1.05 * x1**4 + (1 / 6) * x1**6 + x1 * x2 + x2**2

# Gradient Function for Three-Hump Camel
def three_hump_camel_gradient(x):
    x1, x2 = x
    grad_x1 = 4 * x1 - 4.2 * x1**3 + x1**5 + x2
    grad_x2 = x1 + 2 * x2
    return np.array([grad_x1, grad_x2])

# ADAM Optimization Function with max and min learning rates
def adam_optimization(gradient_func, function, initial_point, max_learning_rate=0.49, min_learning_rate=0.0001, 
                      beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=250, tol=1e-8):
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
            print("Optimization finished after", t, "iterations.")
            break
        
        point = new_point
        history.append(function_value)  # Save function value
        variable_history.append(point)
        
    return point, np.array(history), np.array(variable_history), t

# Initial point for Three-Hump Camel function
initial_point_camel = np.array([-5, 5])

# Run ADAM optimization for Three-Hump Camel function
optimal_point_camel, history_camel, variable_history_camel, iterations = adam_optimization(three_hump_camel_gradient, three_hump_camel, initial_point_camel)

# Plot function value changes over iterations - Three-Hump Camel Function
plt.figure(figsize=(10, 6))
plt.plot(history_camel, label='Three-Hump Camel Function')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Value of Three-Hump Camel Function over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# Plot contour plot of Three-Hump Camel function with optimal path
x1 = np.linspace(-5.5, 5.5, 100)
x2 = np.linspace(-5.5, 5.5, 100)
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, three_hump_camel([X1, X2]), levels=np.linspace(-1, 10, 100))
plt.plot(variable_history_camel[:,0], variable_history_camel[:,1], marker='o', color='green', label='Three-Hump Camel Optimal Path')
plt.plot(initial_point_camel[0], initial_point_camel[1], 'bo')  # Initial point
plt.plot(optimal_point_camel[0], optimal_point_camel[1], 'mo')  # Optimal point - Three-Hump Camel
plt.text(initial_point_camel[0], initial_point_camel[1], 'Initial Point', fontsize=12)
plt.text(optimal_point_camel[0], optimal_point_camel[1], 'Three-Hump Camel Optimal Point', fontsize=12)

plt.title('Contour Plot of Three-Hump Camel Function')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Optimization results
print("Optimization finished after", iterations, "iterations.")
print("Optimal point for Three-Hump Camel function:", optimal_point_camel)
print("Function value at optimal point for Three-Hump Camel:", three_hump_camel(optimal_point_camel))
