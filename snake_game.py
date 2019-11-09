import pygame
from random import randint

BLACK = {0,0,0} # background color
WHITE = (255,255,255) # snake segment color

segment_width = 17
segment_height = 17
segment_margin = 3

class SnakeGame:
    def __init__(self, board_width = 20, board_height = 20):
        print('init')
        self.score = 0
        self.food = []
        self.done = False
        self.board = {'width': board_width, 'height': board_height}

        self.snake_coords = []

        pygame.init()
        self.screen = pygame.display.set_mode([400,400])
        pygame.display.set_caption('Snake')
        self.sprites = pygame.sprite.Group()
        self.clock = pygame.time.Clock()

    def start(self):
        self.snake_init()
        self.place_food([randint(0, self.board['width']), randint(0, self.board['height'])])
        self.render()

    def step(self, key):
        self.create_new_point(key)
        self.check_collisions()
        if self.food == []:
            self.place_food([randint(0, self.board['width']-1), randint(0, self.board['height']-1)])
        else:
            self.remove_last_point()
        self.render()

    def place_food(self,food_coord):
        print('food: ', food_coord[0], food_coord[1])
        self.food = [food_coord[0],food_coord[1]]

    def check_collisions(self):
        head = self.snake_coords[0]
        if (head[0] < 0 or
                head[0] >= self.board['width'] or
                head[1] < 0 or
                head[1] >= self.board['height']):
            print('wall collision')
            pygame.quit()
        if self.snake_coords[0] in self.snake_coords[1:-1]:
            print('snake collision')
        if self.snake_coords[0] == self.food:
            print('food collision')
            self.food = []

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
        for i in range(len(self.snake_coords)):
            rect = pygame.Rect(self.snake_coords[i][0] * (segment_width + segment_margin),
                               self.snake_coords[i][1] * (segment_height + segment_margin),
                               segment_width, segment_height)

            pygame.draw.rect(self.screen, WHITE, rect)
        if self.food != []:
            pygame.draw.rect(self.screen, (255,0,0),
                             (self.food[0] * (segment_width + segment_margin),
                              self.food[1] * (segment_width + segment_margin),
                              segment_width, segment_height))

        pygame.display.update()

    def snake_init(self):
        print('snake_init')

        x = self.board['width']/2
        y = self.board['height']/2

        vertical = randint(0,1) == 0
        for i in range(3):
            point = [x + i, y] if vertical else [x, y + i]
            self.snake_coords.append(point)


if __name__ == "__main__":
    game = SnakeGame()
    game.start()
    currDir = 0
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
