import os
import config
import numpy as np
from pygame.math import Vector2
from snake_game import SNAKE_GAME
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

class DQN:
    def __init__(self, model_file_path, input_shape, output_shape) -> None:
        if model_file_path == None:
            # Create new model to train the agent
            self.__build_new_dqn(input_shape, output_shape)
        else:
            # Load the old_model and start from where left
            self.__load_model(model_file_path)

    def __build_new_dqn(self, input_shape, output_shape):
        self.dqn_model = Sequential([
            Flatten(input_shape=(input_shape)),
            Dense(128, activation='relu'),
            Dense(output_shape, activation='tanh') # (-1, 0, 1)
        ])
        self.dqn_model.compile(
                optimizer = Adam(learning_rate=config.INIT_LR),
                loss='mean_square_error'
              )

    def fit(self, x_train):
        self.dqn_model.fit(x_train, y_train, batch_size, epochs)

    def __load_model(self, model_file_path):
        if not os.path.exists(model_file_path):
            print(model_file_path, 'not exist')
        self.dqn_model = load_model(model_file_path)

    def save(self, model_file_name='sqn_snake.h5'):
        filePath = os.path.join(config.MODEL_PATH, model_file_name)
        self.dqn_model.save(filePath)

class AGENT:
    def __init__(self, model_file_path) -> None:
        self.round_num = 0
        self.model = DQN(model_file_path)

    def get_game_state(self, game):
        food_pos = game.food.position
        snake_sight = config.SNAKE_SIGHT_DISTANCE
        snake_head = game.snake.body[0]
        snake_dir = game.snake.direction

        if snake_dir.x == 0:
            snake_dir_L = Vector2(snake_dir.y, 0)
            snake_dir_R = Vector2(-snake_dir.y, 0)
        elif snake_dir.y == 0:
            snake_dir_L = Vector2(0, -snake_dir.x)
            snake_dir_R = Vector2(0, snake_dir.x)

        state_status = [
            # Snake heading direction_x_y [-1, 0, 1]
            snake_dir.x, 
            snake_dir.y,

            # Danger detection
            game.is_danger(snake_head + snake_sight * snake_dir),   # Danger Ahead
            game.is_danger(snake_head + snake_sight * snake_dir_L),    # Danger Left
            game.is_danger(snake_head + snake_sight * snake_dir_R),    # Danger Right

            # Food censoring
            food_pos.x < snake_head.x,
            food_pos.x > snake_head.x,
            food_pos.y < snake_head.y,
            food_pos.y > snake_head.y
        ]
        # print(state_status)
        return np.array(state_status, dtype=int)

    def get_action(self, state):
        if self.round_num <= config.DISCOVERY_ROUNDS:
            action_list = [
                Vector2(0, -1), # DIR_UP
                Vector2(0, 1), # DIR_DOWN
                Vector2(-1, 0), # DIR_LEFT
                Vector2(1, 0) # DIR_RIGHT
            ]
            rdm_act_idx = np.random.randint(0, 3)
            return action_list[rdm_act_idx]
        else:
            agent_action = self.model.predict(state)
            return np.rint(agent_action[0])

    def q_learning(self, training_status):
        old_game_state = training_status[0]
        agent_action = Vector2(training_status[1], training_status[2])
        round_reward = training_status[3]
        game_over = training_status[4]
        game_score = training_status[5]
        new_game_state = training_status[6]

class AI_TRAINING:
    def __init__(self, surface, model_file_path=None) -> None:
        # Global the game_surface
        global SURFACE
        SURFACE = surface

        self.score_record = 0
        self.agent = AGENT(model_file_path)
        self.game = SNAKE_GAME(surface)
        self.training()

    def training(self):
        while True:
            # Get the game_state
            game_state = self.agent.get_game_state(self.game)

            # Get the agent_action based on the game_state
            agent_action_vector = self.agent.get_action(game_state)
            action_x = agent_action_vector.x
            action_y = agent_action_vector.y

            # Perform the agent_action
            round_reward, game_over, game_score = self.game.agent_play(action_x, action_y)

            # Get new_game_state
            new_game_state = self.agent.get_game_state(self.game)

            # Integrate training_status
            training_status = [
                game_state,
                action_x,
                action_y,
                round_reward,
                game_over,
                game_score,
                new_game_state
            ]

            # Train the agent based on the round_training_status
            self.agent.q_learning(training_status)

            # # Make agent remember the round_training_status
            # self.agent.remember(round_training_status)

            # # If the game is over,
            # if game_over:
            #     # Reset the game environment
            #     self.game.reset_game_state()

            #     # Increment training round number
            #     self.agent.round_num += 1

            #     # Train the agent's long term memory
            #     self.agent.training()

            #     # Print round_score message
            #     print('Round', self.agent.round_num, game_score)

            #     # If break the score_record, print message and save_model
            #     if game_score > self.score_record:
            #         print('Broken Record! New record:', self.score_record)
            #         self.score_record = game_score
            #         self.agent.model.save()

                # Ploting and Analysis

class AI_DEMO:
    def __init__(self, surface, model_file) -> None:
        pass