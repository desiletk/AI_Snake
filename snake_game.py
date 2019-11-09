import pygame
from random import randint
import numpy as np
import math

BLACK = {0,0,0} # background color
WHITE = (255,255,255) # snake segment color

segment_width = 17
segment_height = 17
segment_margin = 3

class SnakeGame:
    def __init__(self, board_width = 20, board_height = 20):
        pygame.init()
        self.snake_coords = []
        self.score = 0
        self.food = []
        self.done = False
        self.board = {'width': board_width, 'height': board_height}
        self.screen = pygame.display.set_mode([400,400])
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

    def start(self):
        self.snake_init()
        self.place_food([randint(0, self.board['width']), randint(0, self.board['height'])])
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
            pygame.quit()

        # check if ate food
        if self.collision_with_food():
            self.place_food([randint(0, self.board['width']-1), randint(0, self.board['height']-1)])
        else:
            self.remove_last_point()

        self.render()

    def place_food(self,food_coord):
        self.food = [food_coord[0],food_coord[1]]

    def collision_with_boundaries(self):
        if (self.snake_coords[0][0] < 0 or self.snake_coords[0][0] >= self.board['width'] or
                self.snake_coords[0][1] < 0 or self.snake_coords[0][1] >= self.board['height']):
            return 1
        else: return 0

    def collision_with_self(self):
        if self.snake_coords[0] in self.snake_coords[1:]: return 1
        else: return 0

    def collision_with_food(self):
        if self.snake_coords[0] == self.food: return 1
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
        pygame.display.update()

    def draw_snake(self):
        for i in range(len(self.snake_coords)):
            rect = pygame.Rect(self.snake_coords[i][0] * (segment_width + segment_margin),
                               self.snake_coords[i][1] * (segment_height + segment_margin),
                               segment_width, segment_height)

            pygame.draw.rect(self.screen, WHITE, rect)

    def draw_food(self):
        if self.food != []:
            pygame.draw.rect(self.screen, (255,0,0),
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
        game.clock.tick(10)
