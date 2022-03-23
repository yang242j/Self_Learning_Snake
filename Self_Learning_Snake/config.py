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
MAX_MEMORY = 10000
BATCH_SIZE = 1000
LEARNING_RATE = 0.00025
DISCOUNT_RATE = 0.95 # < 1
DISCOVERY_ROUND = 100
