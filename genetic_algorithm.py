from snake_game import *
from random import choice, randint, uniform
from test import *

def calc_pop_fitness(pop, generation):
    fitness = []
    for i in range(pop.shape[0]):
        fit = snake_ML(pop[i], generation)
        print('fitness value of chromosome ' + str(i) + ' : ', fit)
        fitness.append(fit)
    return np.array(fitness)

def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999
    return parents


def crossover(parents, offspring_size):
    # creating children for next generation
    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]):

        while True:
            parent1_idx = randint(0, parents.shape[0] - 1)
            parent2_idx = randint(0, parents.shape[0] - 1)
            # produce offspring from two parents if they are different
            if parent1_idx != parent2_idx:
                for j in range(offspring_size[1]):
                    if uniform(0, 1) < 0.5:
                        offspring[k, j] = parents[parent1_idx, j]
                    else:
                        offspring[k, j] = parents[parent2_idx, j]
                break
    return offspring


def mutation(offspring_crossover):
    # mutating the offsprings generated from crossover to maintain variation in the population

    for idx in range(offspring_crossover.shape[0]):
        for _ in range(25):
            i = randint(0, offspring_crossover.shape[1] - 1)

        random_value = np.random.choice(np.arange(-1, 1, step=0.001), size=(1), replace=False)
        offspring_crossover[idx, i] = offspring_crossover[idx, i] + random_value

    return offspring_crossover