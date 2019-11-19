import pygame
from random import randint
import numpy as np
import math
from ff_nn import forward_propagation

BLACK = (0, 0, 0)  # background color
WHITE = (255, 255, 255)  # snake color
RED = (255, 0, 0)  # food color

# keyboard directions
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

SEG_WIDTH = 17
SEG_HEIGHT = 17
SEG_MARGIN = 3

FOOD_SCORE = 500


class Snake:
    def __init__(self, bounds, initial_length=3):
        self.segments = []
        self.score = 0
        self.lifetime = 0
        self.dead = False

        x = bounds['width'] / 2
        y = bounds['height'] / 2
        vertical = randint(0, 1) == 0
        for i in range(initial_length):
            point = [x + i, y] if vertical else [x, y + i]
            self.segments.append(point)

    def move(self, point, ate=False):
        self.lifetime += 1
        self.segments.insert(0, point)
        if ate is False:
            self.remove_last_point()

    def remove_last_point(self):
        self.segments.pop()

    def length(self):
        return len(self.segments)


class Game:
    def __init__(self, board_width=20, board_height=20, use_gui=False, tick_rate=10, title='Snake'):
        pygame.init()
        self.board = {'width': board_width, 'height': board_height}
        self.use_gui = use_gui
        self.tick_rate = tick_rate
        self.snake = Snake(self.board)
        self.food = self.new_food()
        if use_gui:
            pygame.font.init()
            self.screen = pygame.display.set_mode([(SEG_WIDTH+SEG_MARGIN) * self.board['width'] + (SEG_WIDTH+SEG_MARGIN) * 18,
                                                   (SEG_HEIGHT+SEG_MARGIN) * self.board['height'] + (SEG_HEIGHT+SEG_MARGIN) * 2])
            pygame.display.set_caption(title)
            self.clock = pygame.time.Clock()
            self.render()

    def play(self, max_steps=100, weights=None):
        curr_step = 0
        key = 0
        while self.snake.dead is False and curr_step < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN and weights is None:  # allow user to input
                    if event.key == pygame.K_LEFT:
                        key = 0
                    if event.key == pygame.K_RIGHT:
                        key = 1
                    if event.key == pygame.K_UP:
                        key = 2
                    if event.key == pygame.K_DOWN:
                        key = 3
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
            predicted_direction = ''
            if weights is not None:  # predict next move
                angle, v_snake, normalized_v_food, normalized_v_snake = self.angle_with_food()
                front_blocked, left_blocked, right_blocked = self.blocked_direction()

                inputs = np.array([left_blocked, front_blocked, right_blocked,
                                   normalized_v_food[0], normalized_v_food[1],
                                   normalized_v_snake[0], normalized_v_snake[1]]).reshape(-1, 7)

                if self.use_gui:
                    screen = self.screen
                else:
                    screen = None
                predicted_direction = np.argmax(np.array(forward_propagation(np.array(
                    [left_blocked, front_blocked, right_blocked,
                     normalized_v_food[0], normalized_v_snake[0],
                     normalized_v_food[1], normalized_v_snake[1]]).reshape(-1, 7), weights, screen=screen))) - 1

                new_direction = np.array(self.snake.segments[0]) - np.array(self.snake.segments[1])
                if predicted_direction < 0:
                    new_direction = np.array([new_direction[1], -new_direction[0]])
                if predicted_direction > 0:
                    new_direction = np.array([-new_direction[1], new_direction[0]])

                key = self.vector_to_button(new_direction)
            _, _, ate = self.step(key)
            if ate:
                curr_step -= 50
            curr_step += 1
            if self.use_gui:
                self.draw_text('Steps left: ' + str(max_steps - curr_step), 0, 200, WHITE)
                self.draw_text('Time Alive: ' + str(self.snake.lifetime), 0, 210, WHITE)
                self.draw_text('Score     : ' + str(self.snake.score), 0, 220, WHITE)

        return (self.snake.score * 0.75) + (self.snake.lifetime * 0.25)

    def new_food(self, x=None, y=None):
        if x is not None and y is not None:
            return [x.y]
        else:
            while True:
                rdm_pos = [randint(0, self.board['width'] - 1), randint(0, self.board['height'] - 1)]
                if rdm_pos not in self.snake.segments:
                    return rdm_pos

    def step(self, key):
        new_point = self.create_new_point(key)

        hit_bounds, hit_self, hit_food = self.collisions(new_point)

        if hit_food:
            self.snake.score += FOOD_SCORE
            self.food = self.new_food()

        if hit_bounds or hit_self:
            self.snake.dead = True
        else:
            self.snake.move(new_point, ate=hit_food)

        if self.use_gui:
            self.render()

        return hit_bounds, hit_self, hit_food

    def create_new_point(self, key):
        x = y = 0
        if key == 0:
            x = self.snake.segments[0][0] - 1
            y = self.snake.segments[0][1]
        if key == 1:
            x = self.snake.segments[0][0] + 1
            y = self.snake.segments[0][1]
        if key == 2:
            x = self.snake.segments[0][0]
            y = self.snake.segments[0][1] - 1
        if key == 3:
            x = self.snake.segments[0][0]
            y = self.snake.segments[0][1] + 1
        return [x, y]

    def render(self):
        self.screen.fill(BLACK)
        self.draw_snake()
        self.draw_food()
        pygame.draw.line(self.screen, WHITE, [self.board['width'] * (SEG_WIDTH + SEG_MARGIN), 0], [self.board['width'] * (SEG_WIDTH+SEG_MARGIN), self.board['height'] * (SEG_HEIGHT+SEG_MARGIN)])
        pygame.draw.line(self.screen, WHITE, [0, self.board['height'] * (SEG_HEIGHT+SEG_MARGIN)], [self.board['width'] * (SEG_WIDTH + SEG_MARGIN), self.board['height'] * (SEG_HEIGHT + SEG_MARGIN)])
        self.clock.tick(self.tick_rate)
        pygame.display.update()

    def draw_snake(self):
        for i in range(self.snake.length()):
            rect = pygame.Rect(self.snake.segments[i][0] * (SEG_WIDTH + SEG_MARGIN),
                               self.snake.segments[i][1] * (SEG_HEIGHT + SEG_MARGIN),
                               SEG_WIDTH, SEG_HEIGHT)
            pygame.draw.rect(self.screen, WHITE, rect)

    def draw_food(self):
        if self.food is not []:
            pygame.draw.rect(self.screen, RED,
                             (self.food[0] * (SEG_WIDTH + SEG_MARGIN),
                              self.food[1] * (SEG_WIDTH + SEG_MARGIN),
                              SEG_WIDTH, SEG_HEIGHT))

    def draw_text(self, text, x, y, color=WHITE):
        font = pygame.font.SysFont('monospace', 13)
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
        pygame.display.update()

    def angle_with_food(self):
        # direction vectors
        v_food = np.array(self.food) - np.array(self.snake.segments[0])
        v_snake = np.array(self.snake.segments[0]) - np.array(self.snake.segments[1])

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

    def vector_to_button(self, direction):
        if direction.tolist() == [1, 0]:  # right, +x
            return 1
        elif direction.tolist() == [-1, 0]:  # left, -x
            return 0
        elif direction.tolist() == [0, 1]:  # down, +y
            return 3
        else:  # direction.tolist() == [0,-1]  up, -y
            return 2

    def blocked_direction(self):
        v_current = np.array(self.snake.segments[0]) - np.array(self.snake.segments[1])
        v_left = np.array([v_current[1], -v_current[0]])
        v_right = np.array([-v_current[1], v_current[0]])

        front_blocked = self.is_direction_blocked(v_current)
        left_blocked = self.is_direction_blocked(v_left)
        right_blocked = self.is_direction_blocked(v_right)

        return front_blocked, left_blocked, right_blocked

    def is_direction_blocked(self, current_direction_vector):
        next_step = (self.snake.segments[0] + current_direction_vector).tolist()
        hit_bounds, hit_self, _ = self.collisions(next_step)
        return hit_bounds or hit_self

    def collisions(self, point):
        hit_bounds = self.collision_with_bounds(point)
        hit_self = self.collision_with_self(point)
        hit_food = self.collision_with_food(point)

        return hit_bounds, hit_self, hit_food

    def collision_with_bounds(self, point):
        return (point[0] < 0 or point[0] >= self.board['width'] or
                point[1] < 0 or point[1] >= self.board['height'])

    def collision_with_self(self, point):
        return point in self.snake.segments[:-1]

    def collision_with_food(self, point):
        return point == self.food

