import numpy as np
from scipy.optimize import differential_evolution

# Define the objective function to be minimized
def objective_function(x):
    return x[0]**2 + x[1]**2 + 1

# Define bounds for each parameter
bounds = [(-5, 5), (-5, 5)]

# Perform differential evolution optimization
result = differential_evolution(objective_function, bounds)

# Print the result
print('Optimal parameters:', result.x)
print('Objective function value:', result.fun)
