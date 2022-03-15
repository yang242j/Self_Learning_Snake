import config
from snake_game import SNAKE_GAME

class MODEL:
    def __init__(self) -> None:
        pass

class AGENT:
    def __init__(self) -> None:
        self.round_num = 0
        self.model = MODEL()

    def get_game_state(self, game):
        snake_head = game.snake.body[0]

        state = [

        ]

class AI_TRAINING:
    def __init__(self) -> None:
        self.agent = AGENT()
        self.game = SNAKE_GAME()
        self.training()

    def training(self):
        score_record = 0
        while True:
            # Get the game_state
            game_state = self.agent.get_game_state(self.game)

            # Get the agent_action based on the game_state
            agent_action_tuple = self.agent.get_action(game_state)
            action_x = agent_action_tuple[0]
            action_y = agent_action_tuple[1]

            # Perform the agent_action
            round_reward, game_over, game_score = self.game.agent_play(action_x, action_y)

            # Get new_game_state
            new_game_state = self.agent.get_game_state(self.game)

            # Integrate training_status
            round_training_status = [
                game_state,
                agent_action_tuple,
                round_reward,
                new_game_state, 
                game_over
            ]

            # Train the agent based on the round_training_status
            self.agent.round_training(round_training_status)

            # Make agent remember the round_training_status
            self.agent.archive(round_training_status)

            # If the game is over,
            if game_over:
                # Reset the game environment
                self.game.reset_game_state()

                # Increment training round number
                self.agent.num_training += 1

                # Train the agent's long term memory
                self.agent.train_long_memory()

                # If the score_record has been break,
                if game_score > score_record:
                    score_record = game_score
                    # self.agent.model.save()

                print('Round', self.agent.num_training, 'Score', game_score, 'Record', score_record)

                # Ploting and Analysising


class AI_DEMO:
    def __init__(self) -> None:
        pass