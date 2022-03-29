import sys
import config
import random
import pygame
# import numpy as np
from pygame.math import Vector2
# from collections import deque

SURFACE = None

class SNAKE:
    def __init__(self) -> None:
        self.reset()
        self.draw_snake()

    def reset(self):
        # Init or Reset the snake body position
        # x_middle = int(config.CELL_NUMBER/2)
        # y_middle = int(config.CELL_NUMBER/2)
        # y_2 = int(y_middle + 1)
        # y_3 = int(y_middle + 2)
        x_rand = int(random.randint(0, config.CELL_NUMBER))
        y_rand = int(random.randint(0, config.CELL_NUMBER))
        self.body = [
            Vector2(x_rand, y_rand)
            ]
        self.direction = config.DIR_UP
        self.grow = False

    def draw_snake(self):
        for index, block in enumerate(self.body):
            body_block_rect = pygame.Rect(
                int(block.x * config.CELL_SIZE),
                int(block.y * config.CELL_SIZE),
                config.CELL_SIZE,
                config.CELL_SIZE
            )
            pygame.draw.rect(SURFACE, config.SNAKE_COLOR, body_block_rect)

    def move_snake(self):
        self.body.insert(0, self.body[0]+self.direction)
        if self.grow == False:
            self.body = self.body[:-1]
        else:
            self.grow = False

class FOOD:
    def __init__(self, snake_body_list) -> None:
        self.get_random_pos(snake_body_list)

    def get_random_pos(self, snake_body_list):
        new_x = random.randint(0, config.CELL_NUMBER-1)
        new_y = random.randint(0, config.CELL_NUMBER-1)
        self.position = Vector2(new_x, new_y)
        if self.position in snake_body_list:
            self.get_random_pos(snake_body_list)
        else:
            return self.position

    def draw_food(self):
        food_rect = pygame.Rect(
            int(self.position.x * config.CELL_SIZE),
            int(self.position.y * config.CELL_SIZE),
            config.CELL_SIZE,
            config.CELL_SIZE
        )
        pygame.draw.rect(SURFACE, config.FOOD_COLOR, food_rect)

class MAP:
    def __init__(self) -> None:
        SURFACE.fill(config.MAP_COLOR)
        pygame.display.set_caption('The Snake Game')

class SNAKE_GAME:
    def __init__(self, surface, agent=False) -> None:
        # Global the game_surface
        global SURFACE
        SURFACE = surface

        # Init pygame settings
        if agent:
            event_delay = config.AGENT_EVENT_DELAY
        else:
            event_delay = config.PLAYER_EVENT_DELAY
        pygame.time.set_timer(pygame.USEREVENT, event_delay)
        pygame.time.Clock().tick(config.FPS)
        
        # Init the game
        self.map = MAP()
        self.snake = SNAKE()
        self.food = FOOD(self.snake.body)
        self.reset_game_state()

    def reset_game_state(self):
        self.snake.reset()
        self.food.get_random_pos(self.snake.body)
        self.score = 0
        self.reward = 0
        self.health_point = config.SNAKE_CELL_HP * len(self.snake.body)
        self.food_snake_dis = (config.CELL_NUMBER + config.CELL_NUMBER) ** 2
        self.game_over = False

    def user_play(self):
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

        # Return game_over_status & game_score
        return self.game_over, self.score

    def agent_play(self, action_x, action_y):
        # Collect agent input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:   # Quit the game
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN: # User key interaction             
                e_key = event.key
                if e_key == pygame.K_ESCAPE:    # esc -> main_menu
                    stop_training = True
                    return stop_training, self.reward, self.game_over, self.score

        agent_action = Vector2(action_x, action_y)
        # self.action_queue.append(agent_action)
        self.snake.direction = agent_action
        self.update()
        SURFACE.fill(config.MAP_COLOR)
        self.draw_elements()
        pygame.display.update()

        # Return game_over_status & game_score
        stop_training = False
        return stop_training, self.reward, self.game_over, self.score

    def update(self):
        self.health_point -= 1
        self.snake.move_snake()
        self.status_check()

    def draw_elements(self):
        self.food.draw_food()
        self.snake.draw_snake()

    def status_check(self):
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
        else:
            return False
