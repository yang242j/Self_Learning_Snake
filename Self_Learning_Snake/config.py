# Imports
from pygame.math import Vector2

# Define snake speed for different mode
PLAYER_EVENT_DELAY = 150
AGENT_EVENT_DELAY = 10

# Define game map size
FPS = 60
CELL_NUMBER = 20 # Number of cells in both map direction
CELL_SIZE = 40 # How wide or tall each cell is

# Define snake direction 
DIR_UP = Vector2(0, -1)
DIR_DOWN = Vector2(0, 1)
DIR_LEFT = Vector2(-1, 0)
DIR_RIGHT = Vector2(1, 0)

# Define color of game elements
MAP_COLOR = (0, 0, 0)
FOOD_COLOR = (255, 255, 255)
SNAKE_HEAD_COLOR = (215, 0, 0)
SNAKE_BODY_COLOR = (0, 215, 0)
SNAKE_TAIL_COLOR = (0, 0, 215)

# Define training parameters
MODEL_PATH = 'model'
MAX_MEMORY = 1000000
BATCH_SIZE = 500
EPSILON = 1.0
EPSILON_DECAY_RATE = 0.996
EPSILON_MIN = 0.01
AVERAGE_RATE = 0.01
LEARNING_RATE = 0.0005
DISCOUNT_RATE = 0.99 # < 1
