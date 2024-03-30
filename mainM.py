import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Fish:
    def __init__(self, dim):
        self.position = np.random.rand(dim) * 15 - 10
        self.speed = np.random.rand(dim) * 2 - 1
        self.delta_d = np.random.rand(dim) * 2 - 1
        self.delta_c = np.random.rand(dim) * 2 - 1
        self.fitness = float('inf')


class FSS:
    def __init__(self, function, n_fish, max_iter, dim, visual, step_size, food_positions):
        self.function = function
        self.n_fish = n_fish
        self.max_iter = max_iter
        self.dim = dim
        self.visual = visual
        self.step_size = step_size
        self.food_positions = food_positions
        self.best_solution = None

    def initialize_fish(self):
        self.school = [Fish(self.dim) for _ in range(self.n_fish)]

    def evaluate_fitness(self):
        for fish in self.school:
            fish.fitness = self.function(fish.position)

    def move(self):
        for fish in self.school:
            fish.position += fish.speed * self.step_size

    def update_speed(self):
        for fish in self.school:
            fish.speed += fish.delta_d * self.step_size

    def update_deltas(self):
        for fish in self.school:
            if np.random.rand() < self.visual:
                fish.delta_d = np.random.rand(self.dim) * 2 - 1
                fish.delta_c = np.random.rand(self.dim) * 2 - 1

    def attraction(self):
        best_position = min(self.school, key=lambda x: x.fitness).position
        for fish in self.school:
            fish.position += (best_position - fish.position) * 0.01

    def cohesion(self):
        mean_position = np.mean([fish.position for fish in self.school], axis=0)
        for fish in self.school:
            fish.position += (mean_position - fish.position) * 0.01

    def avoidance(self):
        for fish in self.school:
            for other_fish in self.school:
                if fish != other_fish:
                    distance = np.linalg.norm(fish.position - other_fish.position)
                    if distance < 1:
                        fish.position += (fish.position - other_fish.position) * 0.01

    def search(self):
        self.initialize_fish()
        self.evaluate_fitness()
        self.best_solution = min(self.school, key=lambda x: x.fitness)

        fig, ax = plt.subplots()
        ax.set_xlim(-max_size, max_size)
        ax.set_ylim(-max_size, max_size)
        scat = ax.scatter([fish.position[0] for fish in self.school], [fish.position[1] for fish in self.school])
        foods = ax.scatter(self.food_positions[:, 0], self.food_positions[:, 1], c='red', marker='x', label='Food')

        def update(frame_number):
            self.move()
            self.update_speed()
            self.update_deltas()
            self.evaluate_fitness()
            self.attraction()
            self.cohesion()
            self.avoidance()

            for fish in self.school:
                fish.position = np.clip(fish.position, -max_size, max_size)  # Ограничиваем область перемещения

            # Обновляем позиции рыб
            scat.set_offsets([[fish.position[0], fish.position[1]] for fish in self.school])

            # Сохраняем лучшее решение
            self.best_solution = min(self.best_solution, min(self.school, key=lambda x: x.fitness),
                                     key=lambda x: x.fitness)

            return scat,

        ani = animation.FuncAnimation(fig, update, frames=self.max_iter, interval=50, blit=True)
        plt.legend(handles=[foods], loc='upper left')
        ani.save('fss_animation.gif', writer='pillow')

        return self.best_solution.position

# Функция для тестирования - сумма квадратов расстояний до еды
def test_function(position):
    distances = [np.sum((position - food_position) ** 2) for food_position in food_positions]
    return min(distances)

max_size = 1000
# Создаем косяк рыб и местоположение еды
food_positions = np.array([[-500, -500], [-200, -750]])  # Местоположение еды
swarm = [Fish(2) for _ in range(50)]

# Создаем объект FSS
fss = FSS(test_function, n_fish=50, max_iter=600, dim=2, visual=0.3, step_size=0.4, food_positions=food_positions)

# Запускаем алгоритм FSS
print(fss.search())
