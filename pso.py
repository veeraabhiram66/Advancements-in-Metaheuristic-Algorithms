import numpy as np
import random
import math

# Distance Metrics
def manhattan_distance(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2))

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def chebyshev_distance(p1, p2):
    return max(abs(a - b) for a, b in zip(p1, p2))

# PSO Parameters
class ParticleSwarmOptimization:
    def __init__(self, num_particles, dimensions, iterations, distance_func):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.iterations = iterations
        self.distance_func = distance_func
        self.global_best = np.inf
        self.global_best_position = None
        self.particles = [Particle(dimensions, distance_func) for _ in range(num_particles)]

    def optimize(self):
        for _ in range(self.iterations):
            for particle in self.particles:
                particle.evaluate()
                if particle.best_value < self.global_best:
                    self.global_best = particle.best_value
                    self.global_best_position = particle.best_position

            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()

        return self.global_best_position, self.global_best

class Particle:
    def __init__(self, dimensions, distance_func):
        self.position = np.random.permutation(dimensions)
        self.velocity = np.zeros(dimensions)
        self.best_position = self.position.copy()
        self.best_value = np.inf
        self.distance_func = distance_func

    def evaluate(self):
        current_value = self.calculate_distance()
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_position = self.position.copy()

    def calculate_distance(self):
        return sum(self.distance_func(self.position[i], self.position[(i + 1) % len(self.position)]) for i in range(len(self.position)))

    def update_velocity(self, global_best_position):
        inertia = 0.5
        cognitive = 1.5
        social = 1.5
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))

        cognitive_velocity = cognitive * r1 * (self.best_position - self.position)
        social_velocity = social * r2 * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive_velocity + social_velocity

    def update_position(self):
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, 0, len(self.position) - 1)
        self.position = np.argsort(self.position)  # Ensure it's a permutation

# Example usage:
num_cities = 50
coordinates = np.random.rand(num_cities, 2)

# Compute distance matrices
euclidean_distances = np.array([[euclidean_distance(c1, c2) for c2 in coordinates] for c1 in coordinates])

# Run PSO
pso = ParticleSwarmOptimization(num_particles=30, dimensions=num_cities, iterations=100, distance_func=euclidean_distance)
best_path_pso, best_distance_pso = pso.optimize()
print("PSO best path:", best_path_pso)
print("PSO best distance:", best_distance_pso)
