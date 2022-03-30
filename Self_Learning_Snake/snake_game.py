import sys
import config
import random
import pygame
from pygame.math import Vector2

SURFACE = None

class SNAKE:
    """Snake Class
    Defines the initial state of the snake, the look of the snake, as well as how the snake moves
    - SNAKE.reset()
    - SNAKE.draw_snake()
    - SNAKE.move_snake()
    """
    def __init__(self) -> None:
        self.reset()
        self.draw_snake()

    def reset(self) -> None:
        """SNAKE.reset()
        Reset the snake state to random position
        SNAKE.body : list of Vector2 with x,y posions in {integer}
        SNAKE.direction : initial direction is up
        SNAKE.grow : False, {boolean}, whether the snake grows or not in the next update period 
        """
        x_rand = int(random.randint(0, config.CELL_NUMBER-1))
        y_rand = int(random.randint(0, config.CELL_NUMBER-1))
        self.body = [Vector2(x_rand, y_rand)]
        self.direction = config.DIR_UP
        self.grow = False

    def draw_snake(self):
        """SNAKE.draw_snake()
        For each body block, draw the snake
        """
        for index, block in enumerate(self.body):
            body_block_rect = pygame.Rect(
                int(block.x * config.CELL_SIZE),
                int(block.y * config.CELL_SIZE),
                config.CELL_SIZE,
                config.CELL_SIZE
            )
            if index == 0: # Head
                pygame.draw.rect(SURFACE, config.SNAKE_HEAD_COLOR, body_block_rect)
            else: # Snake Body
                pygame.draw.rect(SURFACE, config.SNAKE_BODY_COLOR, body_block_rect)

    def move_snake(self):
        """SNAKE.move_snake()
        Move the sanke by add the direction vector to the head of the snake list and then, 
        if the snake is not growing -> remove the last vector from the list
        if the snake is growing -> keep the list and change the grow status to False
        """
        self.body.insert(0, self.body[0]+self.direction)
        if self.grow == False:
            self.body = self.body[:-1]
        else:
            self.grow = False

class FOOD:
    """ FOOD class
    Define the food position and placement
    - FOOD.get_random_pos(snake_body_list)
    - FOOD.draw_food()
    """
    def __init__(self, snake_body_list) -> None:
        self.get_random_pos(snake_body_list)

    def get_random_pos(self, snake_body_list):
        """FOOD.get_random_pos(snake_body_list)
        Get two random integer from [0, cell_number-1] as the new food position
        if the new position is inside the snake_body, try again
        """
        new_x = int(random.randint(0, config.CELL_NUMBER-1))
        new_y = int(random.randint(0, config.CELL_NUMBER-1))
        self.position = Vector2(new_x, new_y)
        if self.position in snake_body_list:
            self.get_random_pos(snake_body_list)
        else:
            return self.position

    def draw_food(self):
        """FOOD.draw_food()
        Create and draw a food rectangle
        """
        food_rect = pygame.Rect(
            int(self.position.x * config.CELL_SIZE),
            int(self.position.y * config.CELL_SIZE),
            config.CELL_SIZE,
            config.CELL_SIZE
        )
        pygame.draw.rect(SURFACE, config.FOOD_COLOR, food_rect)

class MAP:
    """MAP class
    Define how the map looks
    In this simple map setting, fill the map color and set the caption name.
    """
    def __init__(self) -> None:
        SURFACE.fill(config.MAP_COLOR)
        pygame.display.set_caption('The Snake Game')

class SNAKE_GAME:
    """ SNAKE_GAME class
    Define how the game operates.
    - SNAKE_GAME.reset_game_state()
    - SNAKE_GAME.user_play()
    - SNAKE_GAME.agent_play()
    - SNAKE_GAME.update()
    - SNAKE_GAME.draw_elements()
    - SNAKE_GAME.status_check()
    - SNAKE_GAME.is_danger()
    """
    def __init__(self, surface, agent=False) -> None:
        # Global the game_surface
        global SURFACE
        SURFACE = surface

        # Init pygame settings, event_delay is the game speed
        event_delay = config.AGENT_EVENT_DELAY if agent else config.PLAYER_EVENT_DELAY
        pygame.time.set_timer(pygame.USEREVENT, event_delay)
        pygame.time.Clock().tick(config.FPS)
        
        # Init the game
        self.map = MAP()
        self.snake = SNAKE()
        self.food = FOOD(self.snake.body)
        self.reset_game_state()

    def reset_game_state(self):
        """SNAKE_GAME.reset_game_state()
        Reset the game state:
            - reset snake
            - get new food random posision
            - score = 0
            - reward = 0
            - reset health point
            - reset snake_food_distance (x+y)**2
            - reset game_over status to False
        """
        self.snake.reset()
        self.food.get_random_pos(self.snake.body)
        self.score = 0
        self.reward = 0
        self.health_point = config.SNAKE_CELL_HP * len(self.snake.body)
        self.food_snake_dis = (config.CELL_NUMBER + config.CELL_NUMBER) ** 2
        self.game_over = False

    def user_play(self):
        """SNAKE_GAME.user_play()

        INPUT: None
        
        OUTPUT: game_over, game_score 
        
        Define the operating steps for user interaction
            - Collect user input: 
                - four arrow keys for direction control
                - esc for stop game and back to main_menu
                - update the game when recieves user event
            - fill the surface color
            - draw each elements, snake and food
            - update the pygame display
            - return game_over status and game_score 
        """
        # Collect user inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:   # Quit the game
                pygame.quit()
                sys.exit()
            if event.type == pygame.USEREVENT:
                self.update()
            if event.type == pygame.KEYDOWN: # User key interaction             
                e_key = event.key
                snake_dir = self.snake.direction
                if e_key == pygame.K_UP and snake_dir != config.DIR_DOWN:
                    self.snake.direction = config.DIR_UP
                if e_key == pygame.K_DOWN and snake_dir != config.DIR_UP:
                    self.snake.direction = config.DIR_DOWN
                if e_key == pygame.K_LEFT and snake_dir != config.DIR_RIGHT:
                    self.snake.direction = config.DIR_LEFT
                if e_key == pygame.K_RIGHT and snake_dir != config.DIR_LEFT:
                    self.snake.direction = config.DIR_RIGHT
                if e_key == pygame.K_ESCAPE:    # esc -> main_menu
                    self.game_over = True
                    return self.game_over, self.score

        SURFACE.fill(config.MAP_COLOR)
        self.draw_elements()
        pygame.display.update()
        return self.game_over, self.score

    def agent_play(self, action_x, action_y):
        """SNAKE_GAME.agent_play()
        
        INPUT: 
            - action_x, for x coordinate of agent action
            - action_y, for y coordinate of agent action

        OUTPUT:
            - stop_training {bool}
            - reward {int}
            - game_over {bool}
            - score {int}
        
        Define the steps for agent interacion
            - Collect human user key input
                - esc for stop_training
            - Convert agent action x,y to Vector2
            - set the agent_action as the new direction
            - update game environment
            - fill the game surface to map color
            - draw elements, snake and food
            - update the pygame display
        """
        # Collect agent input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:   # Quit the game
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN: # User key interaction             
                e_key = event.key
                if e_key == pygame.K_ESCAPE:    # esc -> main_menu
                    return True, self.reward, self.game_over, self.score

        agent_action = Vector2(action_x, action_y)
        self.snake.direction = agent_action
        self.update()
        SURFACE.fill(config.MAP_COLOR)
        self.draw_elements()
        pygame.display.update()
        return False, self.reward, self.game_over, self.score

    def update(self):
        """SNAKE_GAME.update()
        steps to take to update the game environment
            - Decrease the snake hp by one
            - move the snake
            - check the status of the snake (health, reward, etc.)
        """
        self.health_point -= 1
        self.snake.move_snake()
        self.status_check()

    def draw_elements(self):
        """SNAKE_GAME.draw_elements()
        steps to take to draw the game
            - draw food
            - draw snake        
        """
        self.food.draw_food()
        self.snake.draw_snake()

    def status_check(self):
        """SNAKE_GAME.status_check()
        - if snake eats a food
            - score++
            - reward = 30
            - grow the snake
            - generate new random food position
            - reset snake hp
        - if any death condition matched, (hit wall, itself, low_hp)
            - reward = -100
            - game over
        - if snake is moving toward the food, reward = 1
        - if snake is moving away from the food, reward = -5
        """
        # snake eat the food
        if self.snake.body[0] == self.food.position:
            self.score += 1
            self.reward = 30
            self.snake.grow = True
            self.food.get_random_pos(self.snake.body)
            self.health_point = config.SNAKE_CELL_HP * len(self.snake.body)

        # check death conditions
        snake_head = self.snake.body[0]
        if self.is_danger(snake_head) or self.health_point==0:
            self.reward = -100
            self.game_over = True

        # Moveing torward food or away from food
        snake_pos = self.snake.body[0]
        food_pos = self.food.position
        new_food_snake_dis = (snake_pos.x - food_pos.x)**2 + (snake_pos.y - food_pos.y)**2
        if new_food_snake_dis < self.food_snake_dis:
            self.reward = 1
        elif new_food_snake_dis > self.food_snake_dis:
            self.reward = -5
        self.food_snake_dis = new_food_snake_dis

    def is_danger(self, pos):
        """SNAKE_GAME.is_danger()
        
        INPUT: vector posion (x, y)
        
        OUTPUT: True/False {bool}

        death condition:
            - hit left wall, x < 0
            - hit top wall, y < 0
            - hit right wall, x > cell_number-1
            - hit bottom wall, y > cell_number-1
            - this position is in snake body list
        
        any death condition matched, return True
        else, return False
        """
        death_cond_list = [
            # pos in the wall
            pos.x < 0,
            pos.y < 0,
            pos.x > config.CELL_NUMBER-1,
            pos.y > config.CELL_NUMBER-1,
            
            # pos in snake body
            pos in self.snake.body[1:]   
        ]
        if any(death_cond_list):
            return True
        return False
