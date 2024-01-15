import numpy as np
import matplotlib.pyplot as plt

#Моментная модификация

coords_x = []
coords_y = []
coords_z = []

def sphere(x):
    return x[0]**2 + x[1]**2

def gradient_descent_with_momentum(start, learning_rate, moment, num_iterations):
    trajectory = []
    previous_update = np.zeros_like(start)
    x = start.copy()

    for _ in range(num_iterations):
        gradient = 2 * x
        update = learning_rate * gradient + moment * previous_update
        x = x - update
        x[2] = sphere(x)

        trajectory.append(x.copy())
        coords_x.append(x[0])
        coords_y.append(x[1])
        coords_z.append(x[2])
        previous_update = update
        if _ % (num_iterations // 10) == 0:
            print(f'Iteration {_}: x = {x[0]}, y = {x[1]}, f(x) = {x[2]}, grad = {gradient}')

    return x, trajectory

start_point = np.array([-9, -9, 0])  
learning_rate = 0.1  
moment = 0.5  
num_iterations = 1000 

optimal_point, trajectory = gradient_descent_with_momentum(start_point, learning_rate, moment, num_iterations)

# Визуализация функции и точки 
x_values = np.linspace(-10, 10, 10)
y_values = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5)

trajectory = np.array(trajectory)
ax.scatter(coords_x, coords_y, coords_z, c='r', marker='o')
ax.scatter(optimal_point[0], optimal_point[1], sphere(optimal_point), c='g', marker='o')

plt.show()


#Адаптивная модификация

coords_x = []
coords_y = []
coords_z = []

def sphere(x):
    return x[0]**2 + x[1]**2

def adam(start, learning_rate, beta1, beta2, epsilon, num_iterations):
    trajectory = []
    m = np.zeros_like(start)
    v = np.zeros_like(start)
    x = start.copy()
    t = 0

    for _ in range(num_iterations):
        t = t + 1
        gradient = 2 * x
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient**2)
        update = learning_rate * m / (np.sqrt(v + epsilon))
        x = x - update
        x[2] = sphere(x)

        trajectory.append(x.copy())
        coords_x.append(x[0])
        coords_y.append(x[1])
        coords_z.append(x[2])
        if _ % (num_iterations // 10) == 0:
            print(f'Iteration {_}: x = {x[0]}, y = {x[1]}, f(x) = {x[2]}, grad = {gradient}')

    return x, trajectory

start_point = np.array([-9, -9, 0])  
learning_rate = 0.1  
num_iterations = 1000  
beta1 = 0.9  
beta2 = 0.999  
epsilon = 1e-8  

optimal_point_adam, trajectory_adam = adam(start_point, learning_rate, beta1, beta2, epsilon, num_iterations)

# Визуализация функции и точки оптимума
x_values = np.linspace(-10, 10, 10)
y_values = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(x_values, y_values)
Z = X**2 + Y**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5)

trajectory_adam = np.array(trajectory_adam)
ax.scatter(coords_x, coords_y, coords_z, c='r', marker='o')
ax.scatter(optimal_point_adam[0], optimal_point_adam[1], sphere(optimal_point_adam), c='g', marker='o')

plt.show()

