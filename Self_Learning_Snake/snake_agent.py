import os
from turtle import update
import config
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from dqn import DEEP_Q_LEARNING_NETWORK
from model import ReplayMemory

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3' 

BATCH_SIZE = config.BATCH_SIZE # @param {type:"integer"}
LR = config.LEARNING_RATE # @param {type:"number"}
GAMMA = config.DISCOUNT_RATE # @param {type:"number"}
DISCOVERY_ROUND = config.DISCOVERY_ROUND # @param {type:"integer"}

class AGENT:
    def __init__(self, game_env, model_file_path=None) -> None:
        # Define instances
        self.game_env = game_env
        self.dqn_trainer = DEEP_Q_LEARNING_NETWORK() 
        if model_file_path != None: self.dqn_trainer = self.dqn_trainer.load_model(model_file_path)

        # Define useful variables
        self.game_record = 0
        self.round_count = 0
        self.stop_training = False

        # Define replay memory pool
        self.memory = ReplayMemory()

    def get_game_state(self):
        snake_sight = 1
        food_pos = self.game_env.food.position
        snake_head = self.game_env.snake.body[0]
        snake_dir = self.game_env.snake.direction

        if snake_dir.x == 0:
            snake_dir_L = config.Vector2(snake_dir.y, 0)
            snake_dir_R = config.Vector2(-snake_dir.y, 0)
        elif snake_dir.y == 0:
            snake_dir_L = config.Vector2(0, -snake_dir.x)
            snake_dir_R = config.Vector2(0, snake_dir.x)
        
        state_status = [
            # Snake heading direction 0, 1, 0, 0
            snake_dir == config.DIR_UP, 
            snake_dir == config.DIR_DOWN,
            snake_dir == config.DIR_LEFT, 
            snake_dir == config.DIR_RIGHT,

            # Danger detection
            self.game_env.is_danger(snake_head + snake_sight * snake_dir),      # Danger Ahead
            self.game_env.is_danger(snake_head + snake_sight * snake_dir_L),    # Danger Left
            self.game_env.is_danger(snake_head + snake_sight * snake_dir_R),    # Danger Right

            # Food censoring
            food_pos.x < snake_head.x,
            food_pos.x > snake_head.x,
            food_pos.y < snake_head.y,
            food_pos.y > snake_head.y
        ]
        return np.array(state_status, dtype=int)

    def get_action(self, state):
        """
        Input: state, boolean list with length of 11
        Output: onehot_action, bool list with length of 3
        1) If the round count is small, means the snake is discovering the game,
        give a random action (epsilon-greedy)
        2) Snake can make three action, go straight, turn left, turn right
            [1, 0, 0] => keep direction, straight
            [0, 1, 0] => turn left
            [0, 0, 1] => turn right
        """
        epsilon = (DISCOVERY_ROUND - self.round_count) / DISCOVERY_ROUND
        if random.randint(0, 1) < epsilon:
            action_idx = random.randint(0, 2)
        else:
            q_pred_list = self.dqn_trainer.predict(state)
            action_idx = tf.math.argmax(q_pred_list).numpy()
        onehot_action = [0, 0, 0]
        onehot_action[action_idx] = 1
        return onehot_action

    def __turn_direction(self, onehot_action):
        """
        Convert onehot action to true direction
            [1, 0, 0] => keep direction, straight
            [0, 1, 0] => turn left
            [0, 0, 1] => turn right
        """
        # Get current snake direction
        snake_dir = self.game_env.snake.direction

        # Get the left and right direction of the current direction
        if snake_dir.x == 0:
            snake_dir_L = config.Vector2(snake_dir.y, 0)
            snake_dir_R = config.Vector2(-snake_dir.y, 0)
        elif snake_dir.y == 0:
            snake_dir_L = config.Vector2(0, -snake_dir.x)
            snake_dir_R = config.Vector2(0, snake_dir.x)

        # Convert based on input onehot action
        if onehot_action == [1, 0, 0]: # Go straight
            new_dir = snake_dir
        elif onehot_action == [0, 1, 0]: # Turn Left
            new_dir = snake_dir_L
        elif onehot_action == [0, 0, 1]: # Turn Right
            new_dir = snake_dir_R
        else:
            print('Get wrong action', onehot_action)
        
        # Return the new direction
        return new_dir

    def play(self, onehot_action):
        new_dir = self.__turn_direction(onehot_action)
        return self.game_env.agent_play(new_dir[0], new_dir[1])

    # def dqn_batch_train(self, batch_size):
    #     """
    #     1) collect traning samples
    #         - If replay memory pool is small, collect all
    #         - If replay memory pool is larger than input batch size, collect random samples
    #     2) send to DQN_trainer to train
    #     """
    #     # 1
    #     sample_batch, new_batch_size = self.memory.random_sample(batch_size=batch_size)

    #     # 2
    #     self.dqn_trainer.train(training_sample=sample_batch, batch_size=new_batch_size)

    def demo(self) -> None:
        # TODO demo play step
        pass

    def training(self) -> None:
        """ AGENT.training()
        training steps:
        1) Get the current game_state
        2) Get decided move_dir based on the game_state
        3) Get the move_reward, is_done status, round_score, by perform the move_dir
        4) Get the next_game_state, by perform the move_dir
        5) Integrate the state_move_sample = [game_state, move_dir, move_reward, next_game_state, is_done]
        6) Train the dqn_model with one sample of game status, state_move_sample
        7) Store/Remember the state_move_sample into pre-defined memory
        8) If the game is_done:
            - Reset game_env
            - If the round_score > game_record:
                - renew the record, save the model
            - Train the dqn_model with a batch of state_move_samples
            - Increment the round_count
            - Print round_info, training_status
        """
        # 1) Get the current game_state
        game_state = self.get_game_state()

        # 2) Get decided move_dir based on the game_state
        onehot_action = self.get_action(game_state)
        
        # 3) Get the move_reward, is_done status, round_score, by perform the move_dir
        self.stop_training, move_reward, is_done, round_score = self.play(onehot_action)
        
        # 4) Get the next_game_state, by perform the move_dir
        next_game_state = self.get_game_state()
        
        # 5) Integrate the state_move_sample = [game_state, move_dir, move_reward, next_game_state, is_done]
        state_move_sample = [game_state, onehot_action, move_reward, next_game_state, is_done]
        
        # 6) Train the dqn_model with one sample of game status, state_move_sample
        self.dqn_trainer.train(training_sample=[state_move_sample], batch_size=1, update=False)
        
        # 7) Store/Remember the state_move_sample into pre-defined memory
        self.memory.append(sample=state_move_sample)
        
        # 8) If snake is dead
        if is_done:
            # Reset game state
            self.game_env.reset_game_state()
            
            # New record, save model
            if round_score > self.game_record:
                print('New record', self.game_record)
                self.game_record = round_score
                self.dqn_trainer.save_model()
            
            # Train DQN model with random batch
            update = True if self.round_count % 10 == 0 else False
            sample_batch, new_batch_size = self.memory.random_sample(batch_size=BATCH_SIZE)
            self.dqn_trainer.train(training_sample=sample_batch, batch_size=new_batch_size, update=update)
            
            # New round
            self.round_count += 1
            print('Round', self.round_count, 'Score', round_score, 'Record', self.game_record)

        # Stop training condition
        if self.stop_training or self.game_record == 2:
            self.stop_training = True

        # Return stop_training bool status, and the record  
        return self.stop_training, self.game_record