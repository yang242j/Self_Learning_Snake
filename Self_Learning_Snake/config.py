# Imports
import os

# Define snake speed for different mode
PLAYER_EVENT_DELAY = 150
AGENT_EVENT_DELAY = 10

# Define game map size
FPS = 60
CELL_NUMBER = 20 # Number of cells in both map direction 
CELL_SIZE = 40 # How wide or tall each cell is 40

# Define color of game elements
MAP_COLOR = (0, 0, 0) #(175, 215, 70)
FOOD_COLOR = (255, 255, 255)
SNAKE_HEAD_COLOR = (215, 0, 0)
SNAKE_BODY_COLOR = (0, 215, 0)
SNAKE_TAIL_COLOR = (0, 0, 215)

# Define training parameters
SNAKE_SIGHT_DISTANCE = 10
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
DISCOVERY_ROUNDS = 10
