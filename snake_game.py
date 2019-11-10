import pygame
from random import randint
import numpy as np
import math

BLACK = {0,0,0} # background color
WHITE = (255,255,255) # snake segment color
RED = (255,0,0) # food color

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

segment_width = 17
segment_height = 17
segment_margin = 3

class SnakeGame:
    def __init__(self, board_width = 20, board_height = 20):
        pygame.init()
        self.snake_coords = list()
        self.score = 0
        self.food = list()
        self.done = False
        self.board = {'width': board_width, 'height': board_height}
        self.screen = pygame.display.set_mode([400,400])
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

    def start(self):
        self.snake_init()
        self.place_food()
        self.render()

    def snake_init(self):
        # start snake in middle of board
        x = self.board['width']/2
        y = self.board['height']/2

        # start snake in random direction
        vertical = randint(0,1) == 0
        for i in range(3):
            point = [x + i, y] if vertical else [x, y + i]
            self.snake_coords.append(point)

    def step(self, key):
        # create new segment in given direction
        self.create_new_point(key)

        # check if snake died
        if self.collision_with_boundaries() or self.collision_with_self():
            #print('QUIT')
            pygame.quit()
            quit()

        # check if ate food
        if self.collision_with_food():
            self.score += 1
            self.place_food()
        else:
            self.remove_last_point()

        self.render()

    def place_food(self):
        while True:
            rdm_pos = [randint(0, self.board['width']), randint(0, self.board['height'])]
            if rdm_pos not in self.snake_coords:
                self.food = rdm_pos
                break

    def collision_with_boundaries(self):
        if (self.snake_coords[0][0] < 0 or self.snake_coords[0][0] >= self.board['width'] or
                self.snake_coords[0][1] < 0 or self.snake_coords[0][1] >= self.board['height']):
            #print('HIT WALL')
            return 1
        else: return 0

    def collision_with_self(self):
        if self.snake_coords[0] in self.snake_coords[1:]:
            #print('HIT SELF')
            return 1
        else: return 0

    def collision_with_food(self):
        if self.snake_coords[0] == self.food:
            #print('ATE FOOD')
            return 1
        else: return 0

    def remove_last_point(self):
        self.snake_coords.pop()

    def create_new_point(self, key):
        x = y = 0
        if key == 0:
            x = self.snake_coords[0][0] - 1
            y = self.snake_coords[0][1]
        if key == 1:
            x = self.snake_coords[0][0] + 1
            y = self.snake_coords[0][1]
        if key == 2:
            x = self.snake_coords[0][0]
            y = self.snake_coords[0][1] - 1
        if key == 3:
            x = self.snake_coords[0][0]
            y = self.snake_coords[0][1] + 1

        self.snake_coords.insert(0, [x,y])

    def render(self):
        self.screen.fill(pygame.Color(0,0,0))
        self.draw_snake()
        self.draw_food()

        # font stuff
        pygame.font.init()
        myfont = pygame.font.SysFont('monospace', 20)
        textsurface = myfont.render("Distance to food: " + str(self.food_distance_from_snake()), True, WHITE)
        self.screen.blit(textsurface,(0,0))
        angle, _, _, _ = self.angle_with_food()
        textsurface = myfont.render("Angle with food: " + str(angle), True, WHITE)
        self.screen.blit(textsurface, (0,20))

        pygame.display.update()

    def draw_snake(self):
        for i in range(len(self.snake_coords)):
            rect = pygame.Rect(self.snake_coords[i][0] * (segment_width + segment_margin),
                               self.snake_coords[i][1] * (segment_height + segment_margin),
                               segment_width, segment_height)

            pygame.draw.rect(self.screen, WHITE, rect)

    def draw_food(self):
        if self.food != []:
            pygame.draw.rect(self.screen, RED,
                             (self.food[0] * (segment_width + segment_margin),
                              self.food[1] * (segment_width + segment_margin),
                              segment_width, segment_height))

    def food_distance_from_snake(self):
        return np.linalg.norm(np.array(self.food) - np.array(self.snake_coords[0]))

    def angle_with_food(self):
        # direction vectors
        v_food = np.array(self.food) - np.array(self.snake_coords[0])
        v_snake = np.array(self.snake_coords[0]) - np.array(self.snake_coords[1])

        # normals of direction vectors
        norm_of_v_food = np.linalg.norm(v_food)
        norm_of_v_snake = np.linalg.norm(v_snake)
        if norm_of_v_food == 0: norm_of_v_food = 10
        if norm_of_v_snake == 0: norm_of_v_snake = 10

        # normalized direction vectors
        normalized_v_food = v_food / norm_of_v_food
        normalized_v_snake = v_snake / norm_of_v_snake

        # angle between snake head and food
        angle = math.atan2(normalized_v_food[1] * normalized_v_snake[0] - normalized_v_food[0] * normalized_v_snake[1],
                           normalized_v_food[1] * normalized_v_snake[1] + normalized_v_food[0] * normalized_v_snake[0]) / math.pi

        return angle, v_snake, normalized_v_food, normalized_v_snake


def blocked_directions(positions, bounds):
    current_direction_vector = np.array(positions[0]) - np.array(positions[1])

    left_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

    front_blocked = is_direction_blocked(positions, current_direction_vector, bounds)
    left_blocked = is_direction_blocked(positions, left_vector, bounds)
    right_blocked = is_direction_blocked(positions, right_vector, bounds)

    return front_blocked, left_blocked, right_blocked


def is_direction_blocked(positions, current_direction_vector, bounds):
    next_step = (positions[0] + current_direction_vector).tolist()
    if next_step[0] < 0 or next_step[0] > bounds[0] or \
            next_step[1] < 0 or next_step[1] > bounds[1] or \
            next_step in positions[:-1]:
        return 1
    else:
        return 0


def generate_random_direction(snake_coords, angle_with_food):
    direction = 0
    if angle_with_food > 0:
        direction = 1
    elif angle_with_food < 0:
        direction = -1

    button_dir = direction_to_button(snake_coords, direction)
    return direction, button_dir


def direction_to_button(positions, desired_direction):
    if positions[0][0] - positions[1][0] < 0: # left
        if desired_direction < 0:
            return DOWN
        elif desired_direction > 0:
            return UP
        else: return LEFT
    elif positions[0][0] - positions[1][0] > 0: # right
        if desired_direction < 0:
            return UP
        elif desired_direction > 0:
            return DOWN
        else: return RIGHT
    elif positions[0][1] - positions[1][1] < 0: # up
        if desired_direction < 0:
            return LEFT
        elif desired_direction > 0:
            return RIGHT
        else: return UP
    elif positions[0][1] - positions[1][1] > 0: # down
        if desired_direction < 0:
            return RIGHT
        elif desired_direction > 0:
            return LEFT
        else: return DOWN


if __name__ == "__main__":
    game = SnakeGame()
    game.start()
    currDir = randint(0,3)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    currDir = 0
                if event.key == pygame.K_RIGHT:
                    currDir = 1
                if event.key == pygame.K_UP:
                    currDir = 2
                if event.key == pygame.K_DOWN:
                    currDir = 3
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
        game.step(currDir)
        game.clock.tick(5)

