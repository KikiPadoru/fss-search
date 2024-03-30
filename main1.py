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
    def __init__(self, function, n_fish, max_iter, dim, visual, step_size):
        self.function = function
        self.n_fish = n_fish
        self.max_iter = max_iter
        self.dim = dim
        self.visual = visual
        self.step_size = step_size
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

    def search(self):
        self.initialize_fish()
        self.evaluate_fitness()
        self.best_solution = min(self.school, key=lambda x: x.fitness)

        for _ in range(self.max_iter):
            self.move()
            self.update_speed()
            self.update_deltas()
            self.evaluate_fitness()
            self.best_solution = min(self.best_solution, min(self.school, key=lambda x: x.fitness),
                                     key=lambda x: x.fitness)

        return self.best_solution.position


def animate_fish_swarm(swarm, food_positions, reaction_distance):
    fig, ax = plt.subplots()
    ax.set_xlim(-22, 22)
    ax.set_ylim(-22, 22)
    scat = ax.scatter([fish.position[0] for fish in swarm], [fish.position[1] for fish in swarm])
    foods = ax.scatter(food_positions[:, 0], food_positions[:, 1], c='red', marker='x', label='Food')
    max_speed = 0.5  # Максимальная скорость рыбы
    random_movement_scale = 0.1  # Масштаб случайного движения

    def update(frame_number):
        for _ in range(1000):
            for fish in swarm:
                # Выбираем ближайшую еду
                nearest_food_index = np.argmin(np.linalg.norm(fish.position - food_positions, axis=1))
                nearest_food = food_positions[nearest_food_index]

                # Вычисляем вектор направления к ближайшей еде
                food_direction = (nearest_food - fish.position)

                # Проверяем, находится ли рыба на расстоянии реакции от еды
                if np.linalg.norm(fish.position - nearest_food) <= reaction_distance:
                    # Если находится, двигаемся прямо к еде с учетом случайного компонента
                    target_direction = (nearest_food - fish.position) / np.linalg.norm(nearest_food - fish.position)
                    random_movement = np.random.randn(2) * random_movement_scale
                    fish.position += (target_direction + random_movement) * max_speed
                else:
                    # Если не находится, двигаемся в общем направлении к еде с учетом случайного компонента
                    target_direction = food_direction / np.linalg.norm(food_direction)
                    random_movement = np.random.randn(2) * random_movement_scale
                    fish.position += (target_direction + random_movement) * max_speed

            # Вычисляем средний вектор направления движения косяка
            swarm_direction = np.mean(
                [(nearest_food - fish.position) for fish, nearest_food in zip(swarm, food_positions)], axis=0)

            # Корректируем общее положение косяка рыб с учетом изменения фитнеса
            for fish in swarm:
                fish.position += swarm_direction * 0.001

        scat.set_offsets([[fish.position[0], fish.position[1]] for fish in swarm])
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)
    plt.legend(handles=[foods], loc='upper left')
    ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)
    # Сохраняем анимацию в виде файла GIF
    ani.save('fss_animation.gif', writer='pillow')


# Создаем косяк рыб и местоположение еды
swarm = [Fish(2) for _ in range(50)]
food_positions = np.array([[-20, -20], [20, -20], [20, 20]])  # Местоположение еды
reaction_distance = 5  # Расстояние реакции на еду
# Запускаем анимацию
animate_fish_swarm(swarm, food_positions, reaction_distance)

