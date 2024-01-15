#Градиентный спуск 1

import numpy as np
import matplotlib.pyplot as plt

coords_x = []
coords_y = []
coords_z = []

def sphere(x):
    return x[0]**2 + x[1]**2

def gradient_descent(start, learning_rate, num_iterations):
    trajectory = []
    x = start.copy()

    for _ in range(num_iterations):
        gradient = 2 * x
        x = x - learning_rate * gradient
        x[2] = sphere(x)
        trajectory.append(x.copy())
        coords_x.append(x[0])
        coords_y.append(x[1])
        coords_z.append(x[2])
        if _ % (num_iterations // 10) == 0:
            print(f'Итерация {_}: x = {x[0]}, y = {x[1]}, f(x) = {x[2]}, grad = {gradient}')

    return x, trajectory

start_point = np.array([-5, -5, 0])  
learning_rate = 0.1  # Скорость обучения
num_iterations = 100  

optimal_point, trajectory = gradient_descent(start_point, learning_rate, num_iterations)

# Визуализация функции и точки оптимума
x = np.linspace(-5, 5, 25)
y = np.linspace(-5, 5, 25)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5)

trajectory = np.array(trajectory)
ax.scatter(coords_x, coords_y, coords_z, c='r', marker='o')
ax.scatter(optimal_point[0], optimal_point[1], sphere(optimal_point), c='g', marker='o')

plt.show()


#Градиентный спуск 2


coords_x = []
coords_y = []
coords_z = []

def rastrigin(x):
    n = len(x)
    A = 10
    return A*n + x[0]**2 - A*np.cos(2*np.pi*x[0]) + x[1]**2 - A*np.cos(2*np.pi*x[1])

def gradient_descent(initial_point, lr, num_iterations):
    point = np.array(initial_point)
    trajectory = [point]
    for _ in range(num_iterations):
        gradient = 2 * point + 20 * np.pi * np.sin(2 * np.pi * point)
        point = point - lr * gradient
        z = rastrigin(point)
        coords_x.append(point[0])
        coords_y.append(point[1])
        coords_z.append(z)
        trajectory.append(point)
        if _ % (num_iterations // 10) == 0:
            print(f'Итерация {_}: x = {point[0]}, y = {point[1]}, f(x) = {z}, grad = {gradient}')
    return point, np.array(trajectory)

# Параметры оптимизации
initial_point = np.array([5, 5])  # начальная точка
lr = 0.01111 # learning rate
num_iterations = 10000000  # количество итераций

optimal_point, trajectory = gradient_descent(initial_point, lr, num_iterations)
print(f'По итогу: x = {optimal_point[0]}, y = {optimal_point[1]}, f(x) = {rastrigin(optimal_point)}')

# Вывод графика Функции Растригина
x_values = np.linspace(-5.12, 5.12, 10)
y_values = np.linspace(-5.12, 5.12, 10)
X, Y = np.meshgrid(x_values, y_values)
Z = 10 * 2 + X**2 - 10*np.cos(2*np.pi*X) + Y**2 - 10*np.cos(2*np.pi*Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5)

ax.scatter(coords_x, coords_y, coords_z, c='r', marker='o')
ax.scatter(optimal_point[0], optimal_point[1], rastrigin(optimal_point), c='g', marker='o')
ax.set_title('Градиентный спуск для функции Растригина')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


#Вычисление погрешности найденного решения в сравнение с аналитическим для нескольких запусков

def compute_error(optimal_point):
    analytical_solution = np.array([0, 0, 0])
    error = np.linalg.norm(optimal_point - analytical_solution)
    return error

num_runs = 5
errors = []

for _ in range(num_runs):
    start_point = np.random.uniform(low=-5, high=5, size=3)
    learning_rate = 0.1  
    num_iterations = 100  
    optimal_point, _ = gradient_descent(start_point, learning_rate, num_iterations)
    error = compute_error(optimal_point)
    errors.append(error)

average_error = np.mean(errors)
print(f"Средняя погрешность: {average_error}")
