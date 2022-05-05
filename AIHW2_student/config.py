#  GAME RESULT CONSTANTS
RESULT_GAME_CONTINUE = 1000
RESULT_PLAYER_DEAD = 1001
RESULT_PLAYER_WON = 1002

# Level constants
GRID_DIMS = 6   # Square grid
FLOOR = 'F'
APPLE = 'A'
WALL = 'W'
SNAKE_BODY = 'B'
SNAKE_HEAD = 'H'
SNAKE_START_COORDINATES = (2, 2)

#  GRAPHIC CONSTANTS
GRID_RENDER_SIZE = (64, 64)
display_width = 800
display_height = 600
game_window_size = (GRID_RENDER_SIZE[0] * (GRID_DIMS + 2), GRID_RENDER_SIZE[1] * (GRID_DIMS + 2))
FPS_MENU = 15
FPS = 30

#  COLOR CONSTANTS
black = (0, 0, 0)
white = (255, 255, 255)
red = (200, 0, 0)
green = (0, 200, 0)
blue = (0, 0, 200)
bright_red = (255, 0, 0)
bright_green = (0, 255, 0)
