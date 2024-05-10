import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock Function
def rosenbrock(x1, x2, a=1, b=100):
    return (a - x1)**2 + b * (x2 - x1**2)**2

# Rosenbrock Function with Constraint
def rosenbrock_with_constraint(x1, x2, a=1, b=100):
    g1 = 1.5 - 0.5 * x1 - x2
    if g1 <= 0:
        return rosenbrock(x1, x2, a, b)
    else:
        raise Exception("Point outside feasible region")

# Booth Function
def booth(x1, x2):
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

# Booth Function with Constraint
def booth_with_constraint(x1, x2):
    g2 = x1**2 + 2*x1 - x2
    if g2 <= 0:
        return booth(x1, x2)
    else:
        raise Exception("Point outside feasible region")

# Three-Hump Camel Function
def three_hump_camel(x1, x2):
    return 2*x1**2 - 1.05*x1**4 + (1/6)*x1**6 + x1*x2 + x2**2

# Three-Hump Camel Function with Constraint
def three_hump_camel_with_constraint(x1, x2):
    if x1**2 + x2**2 <= 1:
        return three_hump_camel(x1, x2)
    else:
        raise Exception("Point outside feasible region")

# ADAM Optimization Function with Constraint
def adam_optimization_with_constraint(gradient_func, function, initial_point, max_learning_rate=0.9, min_learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100, tol=1e-8):
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

        try:
            function_value = function(*new_point)
            history.append(function_value)
            variable_history.append(new_point)
            point = new_point
        except Exception as e:
            print(f"Iteration {t}: {e}")
            break

        if np.linalg.norm(update) < tol:
            print("Optimization converged after", t, "iterations.")
            break

    return point, np.array(history), np.array(variable_history)

# Gradient Functions
def rosenbrock_gradient(x1, x2, a=1, b=100):
    grad_x1 = -2 * (a - x1) - 4 * b * x1 * (x2 - x1**2)
    grad_x2 = 2 * b * (x2 - x1**2)
    return np.array([grad_x1, grad_x2])

def booth_gradient(x1, x2):
    grad_x1 = 2 * (x1 + 2*x2 - 7) + 2 * (2*x1 + x2 - 5) * 2
    grad_x2 = 2 * (x1 + 2*x2 - 7) * 2 + 2 * (2*x1 + x2 - 5)
    return np.array([grad_x1, grad_x2])

def three_hump_camel_gradient(x1, x2):
    grad_x1 = 4 * x1**3 - 4.2 * x1**3 + x1**5 + x2
    grad_x2 = x1 + 2 * x2
    return np.array([grad_x1, grad_x2])

# Initial point
initial_point = np.array([-5, 5])

# Run ADAM optimization for Rosenbrock Function with Constraint
optimal_point_rosenbrock, history_rosenbrock, variable_history_rosenbrock = adam_optimization_with_constraint(rosenbrock_gradient, rosenbrock_with_constraint, initial_point)

# Run ADAM optimization for Booth Function with Constraint
optimal_point_booth, history_booth, variable_history_booth = adam_optimization_with_constraint(booth_gradient, booth_with_constraint, initial_point)

# Run ADAM optimization for Three-Hump Camel Function with Constraint
optimal_point_camel, history_camel, variable_history_camel = adam_optimization_with_constraint(three_hump_camel_gradient, three_hump_camel_with_constraint, initial_point)

# Visualize the contour plot of Rosenbrock function with constraint
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 20, 100)
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, rosenbrock(X1, X2), levels=np.logspace(-1, 3, 20))
plt.plot(variable_history_rosenbrock[:,0], variable_history_rosenbrock[:,1], marker='o', color='red', label='Rosenbrock Optimal Path')
plt.plot(initial_point[0], initial_point[1], 'bo')  # Initial point
plt.plot(optimal_point_rosenbrock[0], optimal_point_rosenbrock[1], 'ro')  # Optimal point - Rosenbrock
plt.axline((2, 0), slope=-2, linestyle='--', color='darkgreen', label='Constraint g1(x1,x2)=0')
plt.text(initial_point[0], initial_point[1], 'Initial Point', fontsize=12)
plt.text(optimal_point_rosenbrock[0], optimal_point_rosenbrock[1], 'Rosenbrock Optimal Point', fontsize=12)
plt.title('Contour Plot of Rosenbrock Function with Constraint')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the contour plot of Booth function with constraint
plt.figure(figsize=(10, 6))
plt.contour(X1, X2, booth(X1, X2), levels=np.logspace(-1, 3, 20))
plt.plot(variable_history_booth[:,0], variable_history_booth[:,1], marker='o', color='blue', label='Booth Optimal Path')
plt.plot(initial_point[0], initial_point[1], 'bo')  # Initial point
plt.plot(optimal_point_booth[0], optimal_point_booth[1], 'ro')  # Optimal point - Booth
plt.axline((0, 0), slope=1, linestyle='--', color='darkgreen', label='Constraint g2(x1,x2)<=0')
plt.text(initial_point[0], initial_point[1], 'Initial Point', fontsize=12)
plt.text(optimal_point_booth[0], optimal_point_booth[1], 'Booth Optimal Point', fontsize=12)
plt.title('Contour Plot of Booth Function with Constraint')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the contour plot of Three-Hump Camel function with constraint
plt.figure(figsize=(10, 6))
plt.contour(X1, X2, three_hump_camel(X1, X2), levels=np.linspace(-1, 10, 100))
plt.plot(variable_history_camel[:,0], variable_history_camel[:,1], marker='o', color='green', label='Three-Hump Camel Optimal Path')
plt.plot(initial_point[0], initial_point[1], 'bo')  # Initial point
plt.plot(optimal_point_camel[0], optimal_point_camel[1], 'mo')  # Optimal point - Three-Hump Camel
plt.axline((0, 0), slope=-1, linestyle='--', color='darkgreen', label='Constraint x1^2 + x2^2<=1')
plt.text(initial_point[0], initial_point[1], 'Initial Point', fontsize=12)
plt.text(optimal_point_camel[0], optimal_point_camel[1], 'Three-Hump Camel Optimal Point', fontsize=12)
plt.title('Contour Plot of Three-Hump Camel Function with Constraint')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Print optimal points
print("Rosenbrock Optimal Point with Constraint:", optimal_point_rosenbrock)
print("Booth Optimal Point with Constraint:", optimal_point_booth)
print("Three-Hump Camel Optimal Point with Constraint:", optimal_point_camel)

# Print function values at optimal points
print("Function Value at Rosenbrock Optimal Point with Constraint:", rosenbrock(*optimal_point_rosenbrock))
print("Function Value at Booth Optimal Point with Constraint:", booth(*optimal_point_booth))
print("Function Value at Three-Hump Camel Optimal Point with Constraint:", three_hump_camel(*optimal_point_camel))
