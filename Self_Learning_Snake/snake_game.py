import sys
import config
import random
import pygame
from pygame.math import Vector2

SURFACE = None
DIR_UP = Vector2(0, -1)
DIR_DOWN = Vector2(0, 1)
DIR_LEFT = Vector2(-1, 0)
DIR_RIGHT = Vector2(1, 0)

class SNAKE:
    def __init__(self) -> None:
        self.reset()
        self.draw_snake()

    def reset(self):
        # Init or Reset the snake body position and the head direction
        self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
        self.direction = DIR_RIGHT
        self.grow = False

    def draw_snake(self):
        for index, block in enumerate(self.body):
            body_block_rect = pygame.Rect(
                int(block.x * config.CELL_SIZE),
                int(block.y * config.CELL_SIZE),
                config.CELL_SIZE,
                config.CELL_SIZE
            )
            if index == 0: # Head
                pygame.draw.rect(SURFACE, config.SNAKE_HEAD_COLOR, body_block_rect)
            elif index == len(self.body)-1: # Tail
                pygame.draw.rect(SURFACE, config.SNAKE_TAIL_COLOR, body_block_rect)
            else: # Snake Body
                pygame.draw.rect(SURFACE, config.SNAKE_BODY_COLOR, body_block_rect)

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
    def __init__(self, surface) -> None:
        # Global the game_surface
        global SURFACE
        SURFACE = surface

        # Init pygame settings
        pygame.time.set_timer(pygame.USEREVENT, config.PLAYER_EVENT_DELAY)
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
                if e_key == pygame.K_UP and snake_dir != DIR_DOWN:
                    self.snake.direction = DIR_UP
                if e_key == pygame.K_DOWN and snake_dir != DIR_UP:
                    self.snake.direction = DIR_DOWN
                if e_key == pygame.K_LEFT and snake_dir != DIR_RIGHT:
                    self.snake.direction = DIR_LEFT
                if e_key == pygame.K_RIGHT and snake_dir != DIR_LEFT:
                    self.snake.direction = DIR_RIGHT
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
                    self.game_over = True
                    return self.game_over, self.score

        self.snake.direction = Vector2(action_x, action_y)
        self.update()
        SURFACE.fill(config.MAP_COLOR)
        self.draw_elements()
        pygame.display.update()

        # Return game_over_status & game_score
        return self.reward, self.game_over, self.score

    def update(self):
        self.snake.move_snake()
        self.status_check()

    def draw_elements(self):
        self.food.draw_food()
        self.snake.draw_snake()

    def is_danger(self, pos):
        death_cond_list = [
            # snake head hits the wall
            pos.x <= 0,
            pos.y <= 0,
            pos.x >= config.CELL_NUMBER,
            pos.y >= config.CELL_NUMBER,
            
            # snake_head hits itself
            pos in self.snake.body[1:]   
        ]
        if any(death_cond_list):
            return True
        else:
            return False

    def status_check(self):
        # snake eat the food
        if self.snake.body[0] == self.food.position:
            self.score += 1
            self.reward += 10
            self.snake.grow = True
            self.food.get_random_pos(self.snake.body)
        
        # check death conditions
        snake_head = self.snake.body[0]
        if self.is_danger(snake_head):
            self.reward -= 10
            self.game_over = True
        # death_cond_list = [
        #     # snake head hits the wall
        #     snake_head.x <= 0,
        #     snake_head.y <= 0,
        #     snake_head.x >= config.CELL_NUMBER,
        #     snake_head.y >= config.CELL_NUMBER,
            
        #     # snake_head hits itself
        #     snake_head in self.snake.body[1:]   
        # ]
        # if any(death_cond_list):
        #     self.reward -= 10
        #     self.game_over = True
        
    
