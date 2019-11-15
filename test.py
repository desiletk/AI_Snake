import tensorflow as tf
from ff_nn import *
import snake_game
from keras.models import model_from_json
import math
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

session = tf.Session(config=config)


def snake_ML(model):
    max_score = 3
    avg_score = 0
    test_games = 1000
    steps_per_game = 2000

    for _ in range(test_games):
        game = snake_game.SnakeGame()
        game.start()
        count_same_direction = 0
        prev_direction = 0

        for _ in range(steps_per_game):
            angle, snake_dir_vector, food_dir_vec_normalized, \
                snake_dir_vector_normalized = game.angle_with_food()

            front_blocked, left_blocked, right_blocked = snake_game.blocked_directions(game.snake_coords, [game.board['width'], game.board['height']])

            predictions = []

            predicted_direction = np.argmax(np.array(model.predict(
                np.array([left_blocked, front_blocked, right_blocked,
                food_dir_vec_normalized[0], snake_dir_vector_normalized[0],
                food_dir_vec_normalized[1], snake_dir_vector_normalized[1],]).reshape(-1,7)))) -1

            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction

            new_direction = np.array(game.snake_coords[0]) - np.array(game.snake_coords[1])
            if predicted_direction < 0:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction > 0:
                new_direction = np.array([-new_direction[1], new_direction[0]])

            button_dir = snake_game.vector_to_button(new_direction)

            next_step = game.snake_coords[0] + snake_dir_vector
            if next_step[0] < 0 or next_step[0] > game.board['width'] or next_step[1] < 0 or next_step[1] > game.board['height']:
                break

            game.step(button_dir)
            game.clock.tick(20)

            #if score > max_score:
            #    max_score = score

        avg_score += max_score

    return max_score, avg_score / test_games


def snake_ML(weights, generation=-1, use_gui=False):
    max_score = 0
    avg_score = 0
    test_games = 1000
    steps_per_game = 500
    score1 = 0
    score2 = 0

    for _ in range(test_games):
        game = snake_game.SnakeGame(generation=generation, use_gui=use_gui)
        game.start()
        count_same_direction = 0
        prev_direction = 0

        for _ in range(steps_per_game):
            angle, snake_dir_vector, food_dir_vec_normalized, \
                snake_dir_vector_normalized = game.angle_with_food()

            front_blocked, left_blocked, right_blocked = snake_game.blocked_directions(game.snake_coords, [game.board['width'], game.board['height']])

            predictions = []

            predicted_direction = np.argmax(np.array(forward_propagation(np.array(
                [left_blocked, front_blocked, right_blocked,
                 food_dir_vec_normalized[0], snake_dir_vector_normalized[0],
                 food_dir_vec_normalized[1], snake_dir_vector_normalized[1]]).reshape(-1, 7), weights))) - 1

            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction
            ''' example for following code
            # direction = [1,0], moving right, want to turn right. 
            # new direction = [0,1]. vector_to_button will return 'down'
            # if continuing straight, new direction = [1,0]. vec to but returns right
            '''
            new_direction = np.array(game.snake_coords[0]) - np.array(game.snake_coords[1])
            if predicted_direction < 0:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction > 0:
                new_direction = np.array([-new_direction[1], new_direction[0]])

            button_dir = snake_game.vector_to_button(new_direction)

            next_step = game.snake_coords[0] + snake_dir_vector
            if (next_step[0] < 0 or next_step[0] > game.board['width'] or
                    next_step[1] < 0 or next_step[1] > game.board['height'] or
                    next_step.tolist() in game.snake_coords[:-1]):
                score1 -= 1000
                break

            game.step(button_dir)
            if use_gui: game.clock.tick(10)

            if game.score > max_score:
                max_score = game.score

            if count_same_direction > 8 and predicted_direction != 0:
                # if its gone the same direction more than 8 times
                # and isnt planning on going straight
                score2 -= 5
            else:
                score2 += 2

        avg_score += max_score
    #print('score: ', game.score, ', score1: ', score1, ' score2: ', score2)
    if len(game.snake_coords) < 10:
        return (math.floor(game.lifetime * game.lifetime * math.pow(2,math.floor(len(game.snake_coords)))))
    else:
        final = game.lifetime * game.lifetime
        final *= math.pow(2,10)
        final *= (len(game.snake_coords)-9)
        return final

def snake_ML_single(weights, generation=-1, use_gui=False):
    max_score = 0
    avg_score = 0
    steps_per_game = 500
    score1 = 0
    score2 = 0

    game = snake_game.SnakeGame(generation=generation, use_gui=use_gui)
    game.start()
    count_same_direction = 0
    prev_direction = 0

    for _ in range(steps_per_game):
        angle, snake_dir_vector, food_dir_vec_normalized, \
            snake_dir_vector_normalized = game.angle_with_food()

        front_blocked, left_blocked, right_blocked = snake_game.blocked_directions(game.snake_coords, [game.board['width'], game.board['height']])

        predictions = []

        predicted_direction = np.argmax(np.array(forward_propagation(np.array(
            [left_blocked, front_blocked, right_blocked,
             food_dir_vec_normalized[0], snake_dir_vector_normalized[0],
             food_dir_vec_normalized[1], snake_dir_vector_normalized[1]]).reshape(-1, 7), weights))) - 1

        if predicted_direction == prev_direction:
            count_same_direction += 1
        else:
            count_same_direction = 0
            prev_direction = predicted_direction
        ''' example for following code
        # direction = [1,0], moving right, want to turn right. 
        # new direction = [0,1]. vector_to_button will return 'down'
        # if continuing straight, new direction = [1,0]. vec to but returns right
        '''
        new_direction = np.array(game.snake_coords[0]) - np.array(game.snake_coords[1])
        if predicted_direction < 0:
            new_direction = np.array([new_direction[1], -new_direction[0]])
        if predicted_direction > 0:
            new_direction = np.array([-new_direction[1], new_direction[0]])

        button_dir = snake_game.vector_to_button(new_direction)

        next_step = game.snake_coords[0] + snake_dir_vector
        if (next_step[0] < 0 or next_step[0] > game.board['width'] or
                next_step[1] < 0 or next_step[1] > game.board['height'] or
                next_step.tolist() in game.snake_coords[:-1]):
            score1 -= 1000
            break

        game.step(button_dir)
        if use_gui: game.clock.tick(10)

        if game.score > max_score:
            max_score = game.score

        if count_same_direction > 8 and predicted_direction != 0:
            # if its gone the same direction more than 8 times
            # and isnt planning on going straight
            score2 -= 5
        else:
            score2 += 2

    avg_score += max_score
    #print('score: ', game.score, ', score1: ', score1, ' score2: ', score2)
    if len(game.snake_coords) < 10:
        return (math.floor(game.lifetime * game.lifetime * math.pow(2,math.floor(len(game.snake_coords)))))
    else:
        final = game.lifetime * game.lifetime
        final *= math.pow(2,10)
        final *= (len(game.snake_coords)-9)
        return final



if __name__ == "__main__":
    json_file = open('model.json', 'r')
    loaded_json_model = json_file.read()
    model = model_from_json(loaded_json_model)
    model.load_weights('model.h5')

    snake_ML(model)
