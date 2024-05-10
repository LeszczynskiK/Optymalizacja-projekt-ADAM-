import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock Function
def rosenbrock(x1, x2, a=1, b=100):
    return (a - x1)**2 + b * (x2 - x1**2)**2

# Booth Function
def booth(x1, x2):
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

# Three-Hump Camel Function
def three_hump_camel(x1, x2):
    return 2*x1**2 - 1.05*x1**4 + (1/6)*x1**6 + x1*x2 + x2**2


# ADAM Optimization Function with Penalty
def adam_optimization_with_penalty(gradient_func, function, constraint_funcs, penalty_factor, initial_point, max_learning_rate=0.9, min_learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100, tol=1e-8):
    # Initialize parameters
    m = np.zeros_like(initial_point, dtype=np.float64)  
    v = np.zeros_like(initial_point, dtype=np.float64)  
    t = 0
    point = initial_point.astype(np.float64)  
    history = []
    variable_history = [initial_point.astype(np.float64)]  

    while t < max_iter:
        t += 1
        gradient = gradient_func(*point)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        learning_rate = max_learning_rate - (max_learning_rate - min_learning_rate) * t / max_iter
        update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Apply penalty for constraints violation
        penalty_gradient = np.zeros_like(point, dtype=np.float64)  
        for constraint_func in constraint_funcs:
            if constraint_func(*point) > 0:
                penalty_gradient += penalty_factor * gradient_func(*point)
        
        new_point = point - update - penalty_gradient

        if np.linalg.norm(update) < tol:
            print("Optimization converged after", t, "iterations.")
            break
        
        point = new_point
        history.append(function(*point, penalty_factor))  # Pass penalty_factor to penalty_objective
        variable_history.append(point)
        
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

# Constraint Functions
def constraint_g1(x1, x2):
    return 1.5 - 0.5 * x1 - x2

def constraint_g2(x1, x2):
    return x1**2 + 2 * x1 - x2

def constraint_g3(x1, x2):
    return x1**2 + x2**2

# Penalty Function
def penalty_objective(x1, x2, r):
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2 + r*(x1 + x2 - 2)**2 + max(0, x1 + x2 - 3)**2

# Initial point
initial_point = np.array([-5, 5])

# Penalty factor
penalty_factor = 30

# Run ADAM optimization for Rosenbrock Function with Penalty
optimal_point_rosenbrock_penalty, history_rosenbrock_penalty, variable_history_rosenbrock_penalty = adam_optimization_with_penalty(rosenbrock_gradient, penalty_objective, [constraint_g1, constraint_g2], penalty_factor, initial_point)

# Run ADAM optimization for Booth Function with Penalty
optimal_point_booth_penalty, history_booth_penalty, variable_history_booth_penalty = adam_optimization_with_penalty(booth_gradient, booth, [constraint_g1, constraint_g2], penalty_factor, initial_point)

# Run ADAM optimization for Three-Hump Camel Function with Penalty
optimal_point_camel_penalty, history_camel_penalty, variable_history_camel_penalty = adam_optimization_with_penalty(three_hump_camel_gradient, three_hump_camel, [constraint_g1, constraint_g2, constraint_g3], penalty_factor, initial_point)

# Visualize function value changes over iterations - Rosenbrock Function
plt.figure(figsize=(10, 6))
plt.plot(history_rosenbrock_penalty, label='Rosenbrock Function with Penalty')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Rosenbrock Function Value vs Iteration')
plt.legend()
plt.grid(True)
plt.show()

# Visualize function value changes over iterations - Booth Function
plt.figure(figsize=(10, 6))
plt.plot(history_booth_penalty, label='Booth Function with Penalty')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Booth Function Value vs Iteration')
plt.legend()
plt.grid(True)
plt.show()

# Visualize function value changes over iterations - Three-Hump Camel Function
plt.figure(figsize=(10, 6))
plt.plot(history_camel_penalty, label='Three-Hump Camel Function with Penalty')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Three-Hump Camel Function Value vs Iteration')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the contour plot of Rosenbrock function with optimal path and constraints
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, rosenbrock(X1, X2), levels=np.logspace(-1, 3, 20))
plt.plot(variable_history_rosenbrock_penalty[:,0], variable_history_rosenbrock_penalty[:,1], marker='o', color='red', label='Rosenbrock Optimal Path with Penalty')
plt.plot(initial_point[0], initial_point[1], 'bo')  
plt.plot(optimal_point_rosenbrock_penalty[0], optimal_point_rosenbrock_penalty[1], 'ro')  
plt.title('Contour Plot of Rosenbrock Function with Constraints and Penalty')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the contour plot of Booth function with optimal path and constraints
plt.figure(figsize=(10, 6))
plt.contour(X1, X2, booth(X1, X2), levels=np.logspace(-1, 3, 20))
plt.plot(variable_history_booth_penalty[:,0], variable_history_booth_penalty[:,1], marker='o', color='blue', label='Booth Optimal Path with Penalty')
plt.plot(initial_point[0], initial_point[1], 'bo')  
plt.plot(optimal_point_booth_penalty[0], optimal_point_booth_penalty[1], 'ro')  
plt.title('Contour Plot of Booth Function with Constraints and Penalty')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the contour plot of Three-Hump Camel function with optimal path and constraints
plt.figure(figsize=(10, 6))
plt.contour(X1, X2, three_hump_camel(X1, X2), levels=np.linspace(-1, 10, 100))
plt.plot(variable_history_camel_penalty[:,0], variable_history_camel_penalty[:,1], marker='o', color='green', label='Three-Hump Camel Optimal Path with Penalty')
plt.plot(initial_point[0], initial_point[1], 'bo')  
plt.plot(optimal_point_camel_penalty[0], optimal_point_camel_penalty[1], 'mo')  
circle_constraint = plt.Circle((0, 0), 1, color='gray', alpha=0.3)
plt.gca().add_patch(circle_constraint)
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.title('Contour Plot of Three-Hump Camel Function with Constraints and Penalty')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Function Value')
plt.legend()
plt.grid(True)
plt.show()

# Print optimal points
print("Rosenbrock Optimal Point with Penalty:", optimal_point_rosenbrock_penalty)
print("Booth Optimal Point with Penalty:", optimal_point_booth_penalty)
print("Three-Hump Camel Optimal Point with Penalty:", optimal_point_camel_penalty)

# Print function values at optimal points
print("Function Value at Rosenbrock Optimal Point with Penalty:", rosenbrock(*optimal_point_rosenbrock_penalty))
print("Function Value at Booth Optimal Point with Penalty:", booth(*optimal_point_booth_penalty))
print("Function Value at Three-Hump Camel Optimal Point with Penalty:", three_hump_camel(*optimal_point_camel_penalty))
