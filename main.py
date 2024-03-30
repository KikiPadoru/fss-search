import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Fish:
    def __init__(self, dimension):
        self.position = np.random.uniform(-100, 100, dimension)  # Инициализация случайной позиции
        self.fitness = float('inf')  # Изначально считаем, что фитнес максимальный

def fss_search(max_iter, num_fish, dimension, objective_function):
    fish_population = [Fish(dimension) for _ in range(num_fish)]
    positions_history = []  # Список для сохранения истории позиций

    for iteration in range(max_iter):
        positions_iteration = []  # Список для хранения позиций на текущей итерации
        for i in range(num_fish):
            current_fish = fish_population[i]
            random_fish_index = np.random.choice([index for index in range(num_fish) if index != i])
            random_fish = fish_population[random_fish_index]
            step_size = np.random.normal(0, 1)
            current_fish.position += step_size * (current_fish.position - random_fish.position)
            current_fish.fitness = objective_function(current_fish.position)
            positions_iteration.append(current_fish.position.copy())  # Сохраняем текущую позицию
        fish_population.sort(key=lambda x: x.fitness)
        positions_history.append([fish.position.copy() for fish in fish_population])  # Сохраняем позиции на текущей итерации

    best_position = fish_population[0].position
    return best_position, positions_history

# Пример использования
def objective_function(x):
    return -sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

# Создаем контур графика функции
x = np.linspace(-20000, 20000, 400)
y = np.linspace(-20000, 20000, 400)
X, Y = np.meshgrid(x, y)
Z = objective_function(np.array([X, Y]))
Z = np.transpose(Z)

best_solution, positions_history = fss_search(max_iter=100, num_fish=30, dimension=2, objective_function=objective_function)
print("Лучшее решение:", best_solution)
print("Значение функции:", objective_function(best_solution))

# Визуализация работы алгоритма
fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.contour(X, Y, Z, levels=20, cmap='jet')  # Контур графика функции
    ax.set_xlim(-20000, 20000)
    ax.set_ylim(-20000, 20000)
    ax.set_title(f"Iteration {frame+1}")
    for fish_positions in positions_history[frame]:
        ax.scatter(fish_positions[0], fish_positions[1], color='blue')
    ax.scatter(best_solution[0], best_solution[1], color='red', label='Лучшее решение', zorder=10)
    ax.legend()

ani = FuncAnimation(fig, update, frames=len(positions_history), interval=150)

# Сохраняем анимацию в виде файла GIF
ani.save('fss_animation.gif', writer='pillow')

