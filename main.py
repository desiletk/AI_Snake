import snake_game
from random import randint

if __name__ == "__main__":
    game = snake_game.SnakeGame()
    game.start()
    for _ in range(20):
        game.step(randint(0,3))
        game.clock.tick(5)
