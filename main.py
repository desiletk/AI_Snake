from genetic_algorithm import *
from tqdm import trange
from snake_game import *
from ff_nn import *
import csv

if __name__ == "__main__":
    sol_per_pop = 100
    num_weights = n_x*n_h + n_h*n_h2 + n_h2*n_y

    pop_size = (sol_per_pop, num_weights)
    new_pop = np.random.choice(np.arange(-1, 1, step=0.01), size=pop_size, replace=True)

    num_gen = 500

    num_parents_mating = 10

    weights = []
    print('### BEGINNING EVOLUTION ###')
    print('Indiv. per population: ', sol_per_pop, ' Generations: ', num_gen, ' Parents per generation: ', num_parents_mating)
    for generation in trange(num_gen):
        #print('\n### GENERATION ' + str(generation) + ' ###')
        fitness = calc_pop_fitness(new_pop)

        #print('### FITTEST CHROMOSOME IN GENERATION ' + str(generation) + ' is having finess value: ' , np.max(fitness))
        parents = select_mating_pool(new_pop, fitness, num_parents_mating)

        offspring_crossover = crossover(parents, (pop_size[0] - parents.shape[0], num_weights))

        offspring_mutation = mutation(offspring_crossover)

        # set new population
        new_pop[0:parents.shape[0], :] = parents
        new_pop[parents.shape[0]:, :] = offspring_mutation

        # only play last generation visible and slow
        if generation == num_gen-1:
            game = Game(use_gui=True, tick_rate=10, board_width=10, board_height=10,
                        title='Best of: Generation ' + str(generation))
        else:
            game = Game(use_gui=False, tick_rate=50, board_width=10, board_height=10,
                        title='Best of: Generation ' + str(generation))

        # find fittest individual
        max_fitness_idx = np.where(new_pop == np.max(new_pop))
        max_fitness_idx = max_fitness_idx[0][0]

        # write weights to file
        if generation == - 1:
            np.savetxt('weights.csv', new_pop[max_fitness_idx], delimiter=',', fmt='%f')

        weights = np.loadtxt('weights.csv')
        # print(weights)
        # play fittest individual
        #score = game.play(weights=new_pop[max_fitness_idx])
        if np.max(fitness) > 10000: break

    print('################## TEST ##################')
    while True:
        print('new test game')
        game = Game(use_gui=True, tick_rate=25, board_width=10,board_height=10, title='Best of the Best')
        game.play(weights=new_pop[max_fitness_idx])
