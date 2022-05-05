import random

import numpy as np
import pygame
from snake_renderer import Renderer
from enum import Enum
from collections import deque

from config import *

import gym
from gym.utils import seeding
from gym import spaces


class SnakeMode(Enum):
    CLASSIC = 0,
    TRON = 1,
    NOTAIL = 2


def convert_direction_to_vector(player_direction):
    if player_direction == 'U':
        return -1, 0
    elif player_direction == 'D':
        return 1, 0
    elif player_direction == 'L':
        return 0, -1
    elif player_direction == 'R':
        return 0, 1
    else:
        return 0, 0


def convert_action_to_direction(action):
    if action == 0:
        return 'U'
    elif action == 1:
        return 'L'
    elif action == 2:
        return 'D'
    elif action == 3:
        return 'R'


def convert_direction_to_action(direction):
    if direction == 'U':
        return 0
    elif direction == 'L':
        return 1
    elif direction == 'D':
        return 2
    elif direction == 'R':
        return 3


def generate_level():
    level_matrix = []
    for i in range(GRID_DIMS + 2):
        level_matrix.append([])
        for j in range(GRID_DIMS + 2):
            # If on edge, place wall
            if i in [0, GRID_DIMS + 1] or j in [0, GRID_DIMS + 1]:
                level_matrix[i].append(WALL)
            else:
                level_matrix[i].append(FLOOR)
    level_matrix[SNAKE_START_COORDINATES[0]][SNAKE_START_COORDINATES[1]] = SNAKE_HEAD

    apple_slots = get_empty_tiles(level_matrix)
    rand_n = random.choice([i for i in range(len(apple_slots))])
    apple_pos = apple_slots[rand_n]

    level_matrix[apple_pos[0]][apple_pos[1]] = APPLE
    return apple_pos, SNAKE_START_COORDINATES, level_matrix


def get_empty_tiles(level_matrix):
    empty_tiles = []
    for r in range(GRID_DIMS + 2):
        for c in range(GRID_DIMS + 2):
            if level_matrix[r][c] == FLOOR:
                empty_tiles.append((r, c))
    return empty_tiles


class Snake(gym.Env):
    def __init__(self, human=False, mode=SnakeMode.NOTAIL, renderer=None,
                 game_window_name="Snake Variants - 5000 in 1"):
        super(Snake, self).__init__()

        self.np_random = 0
        self.game_mode = mode
        self.renderer = renderer

        self.apple_pos = None  # Position for the last spawned apple. Assigned after generate_level() method initially
        self.snake_pos = None  # Position for the head of the snake. Assigned after generate_level() method initially
        self.prev_pos = None
        self.level_matrix = []
        self.snake_direction = 'D'

        self.opposite_directions = {
            'D': 'U',
            'U': 'D',
            'L': 'R',
            'R': 'L'
        }

        self.apple_count = 0
        self.snake_body = deque()
        self.snake_size = 3

        # RL Related variables
        self.done = False
        self.seed()
        self.reward = 0
        self.total_reward = 0
        self.elapsed_steps = 0
        self.total_elapsed_steps = 0
        self.observation_space = spaces.Box(0, 1, shape=(1,))
        self.action_space = spaces.Discrete(4)
        self.state_space = None

        self.hit_the_wall = False
        self.ate_apple = False

        self.human = human

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def move_snake(self):
        # Convert snake direction char to vector
        dx, dy = convert_direction_to_vector(self.snake_direction)

        # Keep the old position
        current_snake_x = self.snake_pos[0]
        current_snake_y = self.snake_pos[1]

        # Calculate the new position
        new_snake_x = self.snake_pos[0] + dx
        new_snake_y = self.snake_pos[1] + dy

        if self.renderer:
            # Update the snake's head sprite
            self.renderer.update_snake_head_sprite(self.snake_direction)

        # Move the snake if the target tile is not wall
        if not self.level_matrix[new_snake_x][new_snake_y] in [WALL, SNAKE_BODY]:
            # If you eat apple, you need to enlarge the snake and respawn the apple
            if self.level_matrix[new_snake_x][new_snake_y] == APPLE:
                self.collect_apple()
                # If it has tail, do not move it but enlarge the list
                if self.game_mode != SnakeMode.NOTAIL:
                    self.snake_body.append((current_snake_x, current_snake_y))
                    self.level_matrix[current_snake_x][current_snake_y] = SNAKE_BODY
                else:
                    self.level_matrix[current_snake_x][current_snake_y] = FLOOR
                self.level_matrix[new_snake_x][new_snake_y] = SNAKE_HEAD

            elif self.game_mode == SnakeMode.CLASSIC:
                # Just move the tail to the prev snake pos
                tail_x, tail_y = self.snake_body.popleft()

                self.level_matrix[tail_x][tail_y] = FLOOR
                self.level_matrix[current_snake_x][current_snake_y] = SNAKE_BODY
                self.level_matrix[new_snake_x][new_snake_y] = SNAKE_HEAD

                self.snake_body.append((current_snake_x, current_snake_y))
                self.snake_size = len(self.snake_body)

            elif self.game_mode == SnakeMode.NOTAIL:
                self.level_matrix[current_snake_x][current_snake_y] = FLOOR
                self.level_matrix[new_snake_x][new_snake_y] = SNAKE_HEAD

            elif self.game_mode == SnakeMode.TRON:
                self.snake_body.append((current_snake_x, current_snake_y))
                self.level_matrix[current_snake_x][current_snake_y] = SNAKE_BODY
                self.level_matrix[new_snake_x][new_snake_y] = SNAKE_HEAD
            self.prev_pos = (current_snake_x, current_snake_y)
            self.snake_pos = (new_snake_x, new_snake_y)
        else:
            self.hit_the_wall = True
            self.done = True

    def collect_apple(self):
        self.apple_count += 1

        apple_slots = get_empty_tiles(self.level_matrix)
        if len(apple_slots) == 0:
            self.done = True
            self.apple_pos = (0, 0)
            return
        rand_n = random.choice([i for i in range(len(apple_slots))])
        self.apple_pos = apple_slots[rand_n]
        self.level_matrix[self.apple_pos[0]][self.apple_pos[1]] = APPLE
        self.ate_apple = True

    def render(self, mode='human'):
        if self.renderer:
            self.renderer.render(self.level_matrix)

    def reset(self):
        self.reward = 0
        self.total_reward = 0
        self.elapsed_steps = 0
        self.done = False

        # Level related variables
        self.apple_pos, self.snake_pos, self.level_matrix = generate_level()
        self.snake_direction = 'D'

        self.apple_count = 0
        self.snake_body = deque()
        if self.game_mode == SnakeMode.NOTAIL:
            self.snake_size = 1
        else:
            # Add 2 parts into the body
            self.snake_size = 3
            self.snake_body.append((1, 1))
            self.snake_body.append((1, 2))
            self.level_matrix[1][1] = SNAKE_BODY
            self.level_matrix[1][2] = SNAKE_BODY

        return self.get_state()

    def is_collision(self, x, y):
        return self.level_matrix[x][y] in [WALL, SNAKE_BODY]

    def is_w_collision(self, x, y):
        return self.level_matrix[x][y] in [WALL]

    def iscurrent_big(self):
        nx, ny = self.snake_pos
        ax, ay = self.apple_pos
        px, py = self.prev_pos

        d1 = abs(nx - ax) + abs(ny - ay)  # current distance
        d2 = abs(px - ax) + abs(py - ay)  # prev distance
        return d1, d2

    def get_state(self):
        return []

    def get_reward(self):
        return 0

    def step(self, action):
        """
        :param action:
            0 --> Go Up
            1 --> Go Left
            2 --> Go Down
            3 --> Go Right
            Need to take a rotation action, perpendicular to your current movement
            Rotation action uses level as a base frame. So when you execute Go Right action, snake will go right if it
            is going up or down, does not matter. If it is going right, Go Right action will do nothing
        :return:
            state, reward, is_done, information (optional)
        """
        self.elapsed_steps += 1
        self.total_elapsed_steps += 1
        self.hit_the_wall = False
        self.ate_apple = False
        directional_action = convert_action_to_direction(action)
        if self.snake_direction != self.opposite_directions[directional_action]:
            self.snake_direction = directional_action
        self.move_snake()

        return self.get_state(), self.get_reward(), self.done, {}

    def quit(self):
        pygame.quit()


if __name__ == "__main__":
    r = Renderer("Test")
    s = Snake(mode=SnakeMode.TRON, renderer=r)
    s.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    result = s.step(3)
                elif event.key == pygame.K_UP:
                    result = s.step(0)
                elif event.key == pygame.K_LEFT:
                    result = s.step(1)
                elif event.key == pygame.K_DOWN:
                    result = s.step(2)
                else:
                    # If no input, move in the current dir
                    result = s.step(convert_direction_to_action(s.snake_direction))

            s.render()
            if s.done:
                pygame.quit()
