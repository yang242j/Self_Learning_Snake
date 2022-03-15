import sys
import config
import pygame
import pygame_menu
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

        # For events of user interaction
        # for event in pygame.event.get():
            
        #     # Quit the game
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         sys.exit()
            
        #     # esc -> main_menu
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_ESCAPE:
        #             main_menu.enable()
        #             main_menu.update(pygame.event.get())
        #             return
        
def agent_training():
    print('Agent Training')

def agent_demo():
    print('Agent Demo')

if __name__ == '__main__':
    # Init the program
    pygame.init()
    MAP_W = config.CELL_SIZE * config.CELL_NUMBER
    MAP_H = config.CELL_SIZE * config.CELL_NUMBER
    main_surface = pygame.display.set_mode((MAP_W, MAP_H))

    # Define fonts
    # game_font = pygame.font.SysFont(None, 25)

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
        title='Start AI Training',
        action=agent_training
    )
    main_menu.add.button(
        title='AI Demo',
        action=agent_demo
    )
    main_menu.add.button(
        title='Quit', 
        action=pygame_menu.events.EXIT
    )

    # Game loop
    main_menu.mainloop(surface=main_surface)
