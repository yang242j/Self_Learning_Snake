# Imports
from pygame.math import Vector2

# Define game event delay for different mode
# higher the number, slower the snake
PLAYER_EVENT_DELAY = 150 # Human player game event delay
AGENT_EVENT_DELAY = 10 # Agent game event delay

# Define game map size
FPS = 60 # Default to 60 Frames per second
CELL_NUMBER = 20 # Number of cells in both map direction
CELL_SIZE = 20 # How wide or tall each cell is

# Define snake direction in 2d-vector, DO NOT CHANGE 
DIR_UP = Vector2(0, -1)
DIR_DOWN = Vector2(0, 1)
DIR_LEFT = Vector2(-1, 0)
DIR_RIGHT = Vector2(1, 0)

# Define color of game elements
MAP_COLOR = (255, 255, 255) # Map color
FOOD_COLOR = (255, 0, 0) # Food color
SNAKE_HEAD_COLOR = (0, 0, 0) # Snake head color : black
SNAKE_BODY_COLOR = (79, 79, 79) # Snake body color : #474747 light grey
SNAKE_CELL_HP = 100 # Health points of each snake cell, long the snake, higher the hp

# Define training parameters
FC_DIM = 256 # Number of hidden nodes in each fully connected hidden layers
EARLY_STOPPING = 100 # Number of round to train without record breaking
MAX_MEMORY = 10000 # How big the memory pool is, number of transition samples
BATCH_SIZE = 64 # How many samples random collected each traning cycle
EPSILON = 1.0 # Start randomness, 1.0 -> 100% random actions 
EPSILON_DECAY_RATE = 0.99 # Randomness decay rate
EPSILON_MIN = 0.01 # The minimum randomness in training
LEARNING_RATE = 0.0025 # ALPHA < 1
DISCOUNT_RATE = 0.99 # GAMMA < 1 
