import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

num_objects = 7
num_intervals_per_day = 6 
num_days = 7  
population_size = 1000
mutation_rate = 0.005
num_generations = 100
object_priorities = np.array([1, 1, 2, 2, 3, 1, 1])

electricity_consumption_data = np.random.rand(num_objects, num_days, num_intervals_per_day)


def generate_population():
    return np.random.randint(2, size=(population_size, num_days * num_intervals_per_day * num_objects))

max_consumption = np.max(electricity_consumption_data)
available_power_scale = 2
available_power = np.random.uniform(0, max_consumption * available_power_scale, size=(num_days, num_intervals_per_day))


def calculate_fitness_with_priorities(population, electricity_consumption, object_priorities, available_power):
    fitness = []

    for ind in population:
        schedule = ind.reshape((num_objects, num_days, num_intervals_per_day))

        total_consumption = np.sum(np.sum(schedule, axis=2), axis=1)

        weighted_total_consumption = total_consumption * object_priorities

        insufficient_power_penalty = np.maximum(0, weighted_total_consumption - available_power.flatten()[:len(weighted_total_consumption)])
        
        # Check if total consumption exceeds available power in each interval and penalize fitness
        consumption_exceeds_power = (total_consumption > available_power.flatten()[:len(total_consumption)])

        # Add penalty for each interval where consumption exceeds available power
        insufficient_power_penalty += np.where(consumption_exceeds_power, total_consumption - available_power.flatten()[:len(total_consumption)], 0)

        fitness.append(np.std(weighted_total_consumption + insufficient_power_penalty))

    return np.array(fitness)

def crossover(parent1, parent2):
    crossover_point1 = np.random.randint(0, len(parent1))
    crossover_point2 = np.random.randint(crossover_point1, len(parent1))

    child1 = np.concatenate((parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    child2 = np.concatenate((parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))

    return child1, child2

def mutate(individual, mutation_rate=mutation_rate):
    mutated_individual = individual.copy()
    mutation_mask = (np.random.rand(*individual.shape) < mutation_rate).astype(int)

    mutated_individual = (mutated_individual + mutation_mask) % 2
    return mutated_individual

def genetic_algorithm_with_priorities(electricity_consumption, object_priorities, available_power, initial_mutation_rate=mutation_rate, mutation_decay_rate=0.001):
    population = generate_population()
    fitness_history = []  

    for generation in range(num_generations):
        # Decay mutation rate linearly
        current_mutation_rate = initial_mutation_rate - mutation_decay_rate * generation
        current_mutation_rate = max(current_mutation_rate, 0.0001)  # Ensure the mutation rate doesn't go below a minimum value

        fitness = calculate_fitness_with_priorities(population, electricity_consumption, object_priorities, available_power)
        fitness_history.append(np.min(fitness))  

        selected_indices = np.argsort(fitness)[:population_size // 2]
        selected_population = population[selected_indices]
        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = selected_population[np.random.choice(len(selected_population), size=2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])

        new_population = [mutate(child, mutation_rate=current_mutation_rate) for child in new_population]

        population = np.array(new_population)

    best_index = np.argmin(calculate_fitness_with_priorities(population, electricity_consumption, object_priorities, available_power))
    best_schedule = population[best_index].reshape((num_objects, num_days, num_intervals_per_day))

    plt.plot(fitness_history)
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()

    return best_schedule

print("\nAvailable Power:")
for day in range(num_days):
    for interval in range(num_intervals_per_day):
        power = available_power[day, interval]
        print(f"Day {day + 1}, Interval {interval + 1}: Power: {power:.2f} kWh")

optimized_schedule_with_priorities = genetic_algorithm_with_priorities(electricity_consumption_data, object_priorities, available_power)

print("\nOptimized Schedule with Priorities:")
print(optimized_schedule_with_priorities)

for day in range(num_days):
    print(f"\nDay {day + 1} Summary:")
    total_electricity = 0
    for obj in range(num_objects):
        print(f"\nObject {obj + 1} Consumption:")
        for interval in range(num_intervals_per_day):
            state = "ON" if optimized_schedule_with_priorities[obj, day, interval] == 1 else "OFF"
            consumption = electricity_consumption_data[obj, day, interval]
            total_electricity += consumption * optimized_schedule_with_priorities[obj, day, interval]
            print(f"Interval {interval + 1}: {state}, Consumption: {consumption:.2f}")

    print(f"\nTotal Electricity Consumption on Day {day + 1}: {total_electricity:.2f} kWh")