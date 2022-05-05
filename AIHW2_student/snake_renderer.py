import pygame
import os
from config import *


class Renderer():
    def __init__(self, game_window_name):
        # Initialize pygame stuff
        pygame.display.init()
        pygame.mixer.init()
        pygame.display.set_caption(game_window_name)
        self.screen = pygame.display.set_mode(game_window_size)
        self.clock = pygame.time.Clock()

        # Render related variables
        ASSET_PATH = os.path.dirname(os.path.abspath(__file__)) + '/assets/'
        self.wall = pygame.transform.scale(pygame.image.load(ASSET_PATH + 'wall.png'), GRID_RENDER_SIZE)
        self.body = pygame.transform.scale(pygame.image.load(ASSET_PATH + 'body.png'), GRID_RENDER_SIZE)
        self.ground = pygame.transform.scale(pygame.image.load(ASSET_PATH + 'ground.png'), GRID_RENDER_SIZE)
        self.apple = pygame.transform.scale(pygame.image.load(ASSET_PATH + 'apple.png'), GRID_RENDER_SIZE)
        self.head_down = pygame.transform.scale(pygame.image.load(ASSET_PATH + 'head.png'), GRID_RENDER_SIZE)
        self.head_up = pygame.transform.flip(pygame.transform.scale(pygame.image.load(ASSET_PATH + 'head.png'), GRID_RENDER_SIZE), False, True)
        self.head_right = pygame.transform.rotate(pygame.transform.scale(pygame.image.load(ASSET_PATH + 'head.png'), GRID_RENDER_SIZE), 90.0)
        self.head_left = pygame.transform.rotate(pygame.transform.scale(pygame.image.load(ASSET_PATH + 'head.png'), GRID_RENDER_SIZE), -90.0)

        self.snake_head_directions = {
            'U': self.head_up,
            'D': self.head_down,
            'L': self.head_left,
            'R': self.head_right
        }

        self.images = {
            WALL: self.wall,
            SNAKE_HEAD: self.head_down,
            SNAKE_BODY: self.body,
            APPLE: self.apple,
            FLOOR: self.ground
        }

    def update_snake_head_sprite(self, snake_direction):
        self.images[SNAKE_HEAD] = self.snake_head_directions[snake_direction]

    def render(self, level_matrix):
        self.clock.tick(FPS)
        pygame.event.get()

        box_size = self.wall.get_width()
        for r in range(GRID_DIMS + 2):
            for c in range(GRID_DIMS + 2):
                self.screen.blit(self.images[level_matrix[r][c]], (c * box_size, r * box_size))
        pygame.display.update()