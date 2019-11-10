import snake_game
import random

def generate_training_data(game):
    training_data_x = []
    training_data_y = []
    training_games = 100
    steps_per_game = 20

    for i in range(training_games):
        # init game
        game.start()

        for _ in range(steps_per_game):
            angle, snake_dir_vector, food_dir_vec_normalized, \
                snake_dir_vector_normalized = game.angle_with_food()

            direction, button_dir = snake_game.generate_random_direction(game.snake_coords, angle)
            front_blocked, left_blocked, right_blocked = snake_game.blocked_directions(game.snake_coords, [game.board['width'], game.board['height']])

            direction, button_dir, training_data_y = generate_training_data_y(game.snake_coords, game.angle_with_food(),
                                                                              button_dir, direction,
                                                                              training_data_y,
                                                                              front_blocked, left_blocked, right_blocked)
            if front_blocked and left_blocked and right_blocked:
                break

            training_data_x.append(
                [left_blocked, front_blocked, right_blocked,
                 food_dir_vec_normalized[0], snake_dir_vector_normalized[0],
                 food_dir_vec_normalized[1], snake_dir_vector_normalized[1]])

            game.step(button_dir)

            game.clock.tick(15)


def generate_training_data_y(positions, angle_with_food, button_dir, direction, training_data_y, front_blocked, left_blocked, right_blocked):
    if direction < 0: # desired direction is left
        if left_blocked is True: # cant go left
            if front_blocked is True and right_blocked is False:
                training_data_y.append([0,0,1])
            elif front_blocked is False and right_blocked is True:
                training_data_y.append([0,1,0])
            elif front_blocked is False and right_blocked is False:
                training_data_y.append([0,0,1])
        else:
                training_data_y.append([1,0,0])

    elif direction > 0:
        if right_blocked is True:
            if front_blocked is True and left_blocked is False:
                training_data_y.append([1,0,0])
            elif front_blocked is False and left_blocked is True:
                training_data_y.append([0,1,0])
            elif front_blocked is False and left_blocked is False:
                training_data_y.append([1,0,0])
        else:
            training_data_y.append([0,0,1])

    else:
        if front_blocked is True:
            if left_blocked is True and right_blocked is False:
                training_data_y.append([0,0,1])
            elif left_blocked is False and right_blocked is True:
                training_data_y.append([1,0,0])
            elif left_blocked is False and right_blocked is False:
                training_data_y.append([0,0,1])
        else:
            training_data_y.append([0,1,0])

    return direction, button_dir, training_data_y


generate_training_data(snake_game.SnakeGame())
