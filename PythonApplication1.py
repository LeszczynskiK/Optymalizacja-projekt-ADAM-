import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x1, x2, a=1, b=100):
    return (a - x1)**2 + b * (x2 - x1**2)**2

def adam_optimization(gradient_func, initial_point, max_learning_rate=0.9, min_learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100, tol=1e-8):
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

        if np.linalg.norm(update) < tol:
            print("Optimization converged after", t, "iterations.")
            break
        
        point = new_point
        history.append(rosenbrock(*point))
        variable_history.append(point)

    return point, np.array(history), np.array(variable_history)

# Gradient of Rosenbrock function
def rosenbrock_gradient(x1, x2, a=1, b=100):
    grad_x1 = -2 * (a - x1) - 4 * b * x1 * (x2 - x1**2)
    grad_x2 = 2 * b * (x2 - x1**2)
    return np.array([grad_x1, grad_x2])

# Initial point
initial_point = np.array([-3, -5])

# Run ADAM optimization
optimal_point, history, variable_history = adam_optimization(rosenbrock_gradient, initial_point)

# Visualize function value changes over iterations
plt.figure(1, figsize=(10, 6))
plt.plot(history, label='Function Value')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Function Value vs Iteration')
plt.legend()
plt.grid(True)

# Visualize the contour plot of Rosenbrock function
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 20, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = rosenbrock(X1, X2)

plt.figure(2, figsize=(10, 6))
plt.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20))

# Plot decision variables for each iteration
for i in range(len(variable_history)):
    plt.plot(variable_history[i][0], variable_history[i][1], marker='o', color=plt.cm.viridis(i / len(variable_history)))

plt.plot(initial_point[0], initial_point[1], 'bo')  # Initial point
plt.plot(optimal_point[0], optimal_point[1], 'ro')  # Optimal point
plt.text(initial_point[0], initial_point[1], 'Initial Point', fontsize=12)
plt.text(optimal_point[0], optimal_point[1], 'Optimal Point', fontsize=12)
plt.title('Contour Plot of Rosenbrock Function')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend(['Optimal Path', 'Initial Point', 'Optimal Point'])
plt.grid(True)

plt.show()

print("Initial Point:", initial_point)
print("Optimal Point:", optimal_point)
print("Function Value at Optimal Point:", rosenbrock(*optimal_point))
