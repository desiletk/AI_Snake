from genetic_algorithm import *
from snake_game import *
from tqdm import trange

if __name__ == "__main__":
    sol_per_pop = 50
    num_weights = n_x*n_h + n_h*n_h2 + n_h2*n_y

    pop_size = (sol_per_pop, num_weights)
    new_pop = np.random.choice(np.arange(-1,1,step=0.01),size=pop_size,replace=True)

    num_gen = 100

    num_parents_mating = 12

    for generation in trange(num_gen):
        print('### GENERATION ' + str(generation) + ' ###')
        fitness = calc_pop_fitness(new_pop, generation)

        print('### FITTEST CHROMOSOME IN GENERATION ' + str(generation) + ' is having finess value: ' , np.max(fitness))
        parents = select_mating_pool(new_pop, fitness, num_parents_mating)

        print(pop_size[0] - parents.shape[0], num_weights)

        offspring_crossover = crossover(parents, (pop_size[0] - parents.shape[0], num_weights))

        offspring_mutation = mutation(offspring_crossover)

        new_pop[0:parents.shape[0], :] = parents
        new_pop[parents.shape[0]:, :] = offspring_mutation

    print('################## TEST ##################')
    print(new_pop[0])
    snake_ML(new_pop[0], use_gui=True)