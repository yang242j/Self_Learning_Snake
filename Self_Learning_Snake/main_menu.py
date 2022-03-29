import config
import pygame
import pygame_menu
from snake_ddqn_agent import DDQN_Agent
from snake_game import SNAKE_GAME

def play_snake_game():
    print('Snake Game start')
    
    # Disable and reset the pygame_menu
    main_menu.disable()
    main_menu.full_reset()

    # Create a game surface
    game_surface = pygame.Surface((MAP_W, MAP_W))
    
    # Init the snake game
    game = SNAKE_GAME(game_surface)
    
    # Game loop
    while True:
        # Built and flip the game_surface onto the main_surface
        main_surface.blit(game_surface, (0, 0))
        pygame.display.flip()

        # Play user controlled game
        game_over, game_score = game.user_play()

        if game_over == True:
            print('Round Score:', game_score)
            main_menu.enable()
            main_menu.update(pygame.event.get())
            return

def agent_training(model_file_path):
    if model_file_path == None:
        print('New Agent Training')
    else:
        print('Resume Agent Training')

    # Disable and reset the pygame_menu
    main_menu.disable()
    main_menu.full_reset()

    # Create an agent surface
    agent_surface = pygame.Surface((MAP_W, MAP_W))
    
    # Init the game and agent
    game_env = SNAKE_GAME(agent_surface, agent=True)
    agent = DDQN_Agent(game_env, model_file_path)

    # Training loop
    while True:
        # Built and flip the game_surface onto the main_surface
        main_surface.blit(agent_surface, (0, 0))
        pygame.display.flip()

        # Start the agent training
        stop_training, game_record = agent.trainer()

        # Training END, return to main_menu
        if stop_training == True:
            print('Training Record:', game_record)
            main_menu.enable()
            main_menu.update(pygame.event.get())
            return

def agent_demo(model_file_path):
    # Disable and reset the pygame_menu
    main_menu.disable()
    main_menu.full_reset()

    print('Agent Demo')

    # Create a demo surface
    demo_surface = pygame.Surface((MAP_W, MAP_W))

    # Init the game and agent
    game_env = SNAKE_GAME(demo_surface, agent=False)
    agent = DDQN_Agent(game_env, model_file_path)

    

    while model_file_path:
        # Built and flip the game_surface onto the main_surface
        main_surface.blit(demo_surface, (0, 0))
        pygame.display.flip()

        # Start demo
        end_demo, demo_record = agent.demo()

        # End demo
        if end_demo == True:
            print('Highest Demo Record:', demo_record)
            main_menu.enable()
            main_menu.update(pygame.event.get())
            return

if __name__ == '__main__':
    # Init the program
    pygame.init()
    MAP_W = config.CELL_SIZE * config.CELL_NUMBER
    MAP_H = config.CELL_SIZE * config.CELL_NUMBER
    main_surface = pygame.display.set_mode((MAP_W, MAP_H))

    # Define main menu 
    main_menu = pygame_menu.Menu(
        title='Self-Learning Snake',
        width=MAP_W, 
        height=MAP_H,
        theme=pygame_menu.themes.THEME_SOLARIZED
    )
    main_menu.add.button(
        title='Play',
        action=play_snake_game
    )
    main_menu.add.button(
        'New AI Training',
        agent_training,
        None
    )
    main_menu.add.button(
        'Resume AI Training',
        agent_training,
        'model/ddqn_snake.h5'
    )
    main_menu.add.button(
        'AI Demo',
        agent_demo,
        'model/ddqn_snake.h5'
    )
    main_menu.add.button(
        title='Quit', 
        action=pygame_menu.events.EXIT
    )

    # Game loop
    main_menu.mainloop(surface=main_surface)
