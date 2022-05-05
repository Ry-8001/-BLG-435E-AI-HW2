import pygame
from snake_environment import *
from snake_renderer import *
import numpy as np


class CustomSnake(Snake):
    def __init__(self,
                 human=False,
                 mode=SnakeMode.NOTAIL,
                 render=True,
                 game_window_name="Snake Variants - 5000 in 1"):
        super(CustomSnake, self).__init__(human, mode, Renderer(game_window_name) if render else None)

        """
        DEFINE YOUR OBSERVATION SPACE DIMENSIONS HERE FOR EACH MODE.
        JUST CHANGING THE "obs_space_dim" VARIABLE SHOULD BE ENOUGH
        """
        if mode == SnakeMode.NOTAIL:
            obs_space_dim = 12
            self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))
        elif mode == SnakeMode.CLASSIC:
            obs_space_dim = 11
            self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))
        elif mode == SnakeMode.TRON:
            obs_space_dim = 11
            self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))

    def get_state(self):
        """
        Define your state representation here
        :return:
        """
        # snake pos
        head_x, head_y = self.snake_pos
        # apple pos
        app_x, app_y = self.apple_pos

        # take left
        dxl, dyl = convert_direction_to_vector('L')
        point_l_x, point_l_y = (head_x + dxl, head_y + dyl)
        # take right
        dxr, dyr = convert_direction_to_vector('R')
        point_r_x, point_r_y = (head_x + dxr, head_y + dyr)
        # take up
        dxu, dyu = convert_direction_to_vector('U')
        point_u_x, point_u_y = (head_x + dxu, head_y + dyu)
        # take down
        dxd, dyd = convert_direction_to_vector('U')
        point_d_x, point_d_y = (head_x + dxd, head_y + dyd)

        # look at current direction
        dir_l = self.snake_direction == 'L'
        dir_r = self.snake_direction == 'R'
        dir_u = self.snake_direction == 'U'
        dir_d = self.snake_direction == 'D'
        if self.game_mode == SnakeMode.NOTAIL:

            state = [
                # danger straight (Wall)
                int((dir_r and self.is_w_collision(point_r_x, point_r_y)) or
                    (dir_l and self.is_w_collision(point_l_x, point_l_y)) or
                    (dir_u and self.is_w_collision(point_u_x, point_u_y)) or
                    (dir_d and self.is_w_collision(point_d_x, point_d_y))),
                # danger right (Wall)
                int((dir_u and self.is_w_collision(point_r_x, point_r_y)) or
                    (dir_d and self.is_w_collision(point_l_x, point_l_y)) or
                    (dir_l and self.is_w_collision(point_u_x, point_u_y)) or
                    (dir_r and self.is_w_collision(point_d_x, point_d_y))),

                # danger left (Wall)
                int((dir_d and self.is_w_collision(point_r_x, point_r_y)) or
                    (dir_u and self.is_w_collision(point_l_x, point_l_y)) or
                    (dir_r and self.is_w_collision(point_u_x, point_u_y)) or
                    (dir_l and self.is_w_collision(point_d_x, point_d_y))),

                # danger is down (Wall)
                int((dir_r and self.is_w_collision(point_l_x, point_l_y)) or
                    (dir_l and self.is_w_collision(point_r_x, point_r_y)) or
                    (dir_u and self.is_w_collision(point_d_x, point_d_y)) or
                    (dir_d and self.is_w_collision(point_u_x, point_u_y))),

                # move direction
                int(dir_l),
                int(dir_r),
                int(dir_u),
                int(dir_d),

                # food loc

                int(app_y < head_y),  # l
                int(app_y > head_y),  # r
                int(app_x < head_x),  # u
                int(app_x > head_x)  # d

            ]
            return state
        elif self.game_mode == SnakeMode.CLASSIC or self.game_mode == SnakeMode.TRON:

            state = [
                # danger straight (Body or Wall)
                int((dir_r and self.is_collision(point_r_x, point_r_y)) or
                    (dir_l and self.is_collision(point_l_x, point_l_y)) or
                    (dir_u and self.is_collision(point_u_x, point_u_y)) or
                    (dir_d and self.is_collision(point_d_x, point_d_y))),
                # danger right (Body or Wall)
                int((dir_u and self.is_collision(point_r_x, point_r_y)) or
                    (dir_d and self.is_collision(point_l_x, point_l_y)) or
                    (dir_l and self.is_collision(point_u_x, point_u_y)) or
                    (dir_r and self.is_collision(point_d_x, point_d_y))),

                # danger left (Body or Wall)
                int((dir_d and self.is_collision(point_r_x, point_r_y)) or
                    (dir_u and self.is_collision(point_l_x, point_l_y)) or
                    (dir_r and self.is_collision(point_u_x, point_u_y)) or
                    (dir_l and self.is_collision(point_d_x, point_d_y))),

                # move dir
                int(dir_l),
                int(dir_r),
                int(dir_u),
                int(dir_d),

                # food loc

                int(app_y < head_y),  # l
                int(app_y > head_y),  # r
                int(app_x < head_x),  # u
                int(app_x > head_x)  # d

            ]
            return state
        else:
            raise ModuleNotFoundError("This mode is currently not supported. Please refer to the manual")

    def get_reward(self):
        if self.game_mode == SnakeMode.NOTAIL:
            # if snake eats the apple
            if self.ate_apple:
                self.reward = 1
                self.total_reward += self.reward
                return self.reward
            # if snake hits the wall (Game over)
            elif self.hit_the_wall:
                self.reward = -1
                self.total_reward += self.reward
                return self.reward
            # if it did not change its pos
            elif self.prev_pos == self.snake_pos:
                self.reward = -0.1
                self.total_reward += self.reward
                return self.reward
            # Now look if it is getting closer to the apple
            else:
                new_distance, prev_distance = self.iscurrent_big()
                # distance is increasing
                if new_distance >= prev_distance:
                    self.reward = - 0.1
                    self.total_reward += self.reward
                    return self.reward
                # distance is decreasing
                else:
                    self.reward = 0.1
                    self.total_reward += self.reward
                    return self.reward
        elif self.game_mode == SnakeMode.CLASSIC or self.game_mode == SnakeMode.TRON:
            if self.ate_apple:
                self.reward = 1
                self.total_reward += self.reward
                return self.reward
            elif self.hit_the_wall:
                self.reward = -1
                self.total_reward += self.reward
                return self.reward
            # eats own body
            elif self.done:
                self.reward = -1
                self.total_reward += self.reward
                return self.reward
            elif self.prev_pos == self.snake_pos:
                self.reward = -0.1
                self.total_reward += self.reward
                return self.reward
            else:
                new_distance, prev_distance = self.iscurrent_big()
                if new_distance >= prev_distance:
                    self.reward = - 0.1
                    self.total_reward += self.reward
                    return self.reward
                else:
                    self.reward = 0.1
                    self.total_reward += self.reward
                    return self.reward
        else:
            raise ModuleNotFoundError("This mode is currently not supported. Please refer to the manual")
