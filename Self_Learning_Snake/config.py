# Imports
from pygame.math import Vector2

# Define snake speed for different mode
PLAYER_EVENT_DELAY = 150
AGENT_EVENT_DELAY = 10

# Define game map size
FPS = 60
CELL_NUMBER = 20 # Number of cells in both map direction
CELL_SIZE = 20 # How wide or tall each cell is

# Define snake direction 
DIR_UP = Vector2(0, -1)
DIR_DOWN = Vector2(0, 1)
DIR_LEFT = Vector2(-1, 0)
DIR_RIGHT = Vector2(1, 0)

# Define color of game elements
MAP_COLOR = (255, 255, 255)
FOOD_COLOR = (255, 0, 0)
SNAKE_HEAD_COLOR = (0, 0, 0) # color : black
SNAKE_BODY_COLOR = (79, 79, 79) # color : #474747
SNAKE_CELL_HP = 100

# Define training parameters
# MODEL_PATH = 'model'
MAX_MEMORY = 10000
BATCH_SIZE = 64
EPSILON = 1.0
EPSILON_DECAY_RATE = 0.99
EPSILON_MIN = 0.01
LEARNING_RATE = 0.0025 # ALPHA < 1
DISCOUNT_RATE = 0.99 # GAMMA < 1 
