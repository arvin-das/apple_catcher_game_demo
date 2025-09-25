import pygame
import sys
import random
import time
import numpy as np
from constants import *
from preprocessing import *
from classification import *
from data_collection import *

# When using the second lab screen

#import os
#os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d" % (1920+960,-480)

class Apple_catcher_game:
    def __init__(self):
        self.subject_number = input("Enter the subject number: ")
        while self.subject_number.isdigit() == False:
            print("\nSubject number must be an integer!")
            self.subject_number = input("Enter the subject number: ")

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + LOAD_BAR_HEIGHT))
        pygame.display.set_caption("Apple Catcher Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)

        # Time variables
        self.before_marker_time = BEFORE_MARKER_TIME
        self.marker_time = MARKER_TIME
        self.total_time = TOTAL_TIME
        self.start_time = time.time()

        # Game variables
        self.end_value = END_VALUE # Default end value of 20
        self.player_pos = [SCREEN_WIDTH//2, SCREEN_HEIGHT - PLAYER_HEIGHT]

        self.apple_distribution =[1]*(self.end_value//2) + [0]*(self.end_value//2)
        random.shuffle(self.apple_distribution)
        self.apple_distribution.append(-1) # To increase the length so we dont access out of bounds
        self.apple_pos = self.get_random_starting_position()

        self.apple_speed = (SCREEN_HEIGHT) / (self.total_time*FPS)
        self.load_bar_speed = SCREEN_WIDTH / (self.total_time*FPS)
        self.score = 0
        self.failures = 0
        self.right_hand = "closed"
        self.left_hand = "closed"
        self.game_mode = "training"

        # Load and scale images
        self.tree_image = pygame.image.load(TREE_IMAGE_PATH)
     
        width = int(self.tree_image.get_width() * 3)
        height = int(self.tree_image.get_height() * 2)
        self.tree_image = pygame.transform.scale(self.tree_image, (width, height))
        self.left_hand_open = self.load_and_scale_image(LEFT_HAND_OPEN_PATH, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.left_hand_closed = self.load_and_scale_image(LEFT_HAND_CLOSED_PATH, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.right_hand_open = self.load_and_scale_image(RIGHT_HAND_OPEN_PATH, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.right_hand_closed = self.load_and_scale_image(RIGHT_HAND_CLOSED_PATH, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.apple_image = self.load_and_scale_image(APPLE_IMAGE_PATH, APPLE_SIZE, APPLE_SIZE)

        # Marker position
        self.first_marker = SCREEN_WIDTH * self.before_marker_time /self.total_time
        self.second_marker= SCREEN_WIDTH * (self.before_marker_time + self.marker_time)/self.total_time

        # Initialize the data stream
        self.inlet = create_lsl_inlet(STREAM_NAME)
        self.inlet_info = create_mne_info(self.inlet)
        self.offset = self.inlet.time_correction()

        # Flags
        self.classifier_done = False

        # Saving data
        self.epochs = []
        self.X = []
        self.Y = []
        self.predictions = []
        
        # Classifier
        self.classifier, self.inverse_operator = initialize_from_training_data(self.subject_number)
   
    def load_and_scale_image(self, image_path, width, height):
        image = pygame.image.load(image_path)
        image = pygame.transform.scale(image, (width, height))
        return image
        
    def get_random_starting_position(self):
        side = self.apple_distribution[0]
        self.apple_distribution.pop(0)
        if side == 0:
            pos_x = self.player_pos[0]-APPLE_SIZE-PLAYER_WIDTH//2
        else:
            pos_x = self.player_pos[0]+PLAYER_WIDTH//2-10
        pos_y = 0
        return [pos_x, pos_y]
    
    def draw(self):
        # Set background color
        self.screen.fill(BACKGROUND_COLOR)

         # Apple
        self.screen.blit(self.apple_image, (self.apple_pos[0], self.apple_pos[1]))

        # Background tree
        self.screen.blit(self.tree_image, (SCREEN_WIDTH/2+5-self.tree_image.get_width()/2, 0))

        # Player hands
        self.screen.blit(getattr(self, f"left_hand_{self.left_hand}"), (self.player_pos[0] - PLAYER_WIDTH, self.player_pos[1]))
        self.screen.blit(getattr(self, f"right_hand_{self.right_hand}"), (self.player_pos[0], self.player_pos[1]))
        
        # Scoreboard
        score_text = self.font.render(f'Score: {self.score}', True, (0, 0, 0))
        failures_text = self.font.render(f'Failures: {self.failures}', True, (255, 0, 0))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(failures_text, (10, 50))

        # Load bar
        load_bar_width = SCREEN_WIDTH * self.elapsed_time / TOTAL_TIME
        load_bar_rect = pygame.Rect(0, SCREEN_HEIGHT, load_bar_width, LOAD_BAR_HEIGHT)
        pygame.draw.rect(self.screen, LOAD_BAR_COLOR, load_bar_rect)

        # Markers
        first_marker_rect = pygame.Rect(self.first_marker, SCREEN_HEIGHT, 2, LOAD_BAR_HEIGHT)
        pygame.draw.rect(self.screen, MARKER_BAR_COLOR, first_marker_rect)
        second_marker_rect = pygame.Rect(self.second_marker, SCREEN_HEIGHT, 2, LOAD_BAR_HEIGHT)
        pygame.draw.rect(self.screen, MARKER_BAR_COLOR, second_marker_rect)

    def reset_for_next_apple(self):
        self.apple_pos = self.get_random_starting_position()
        self.start_time = time.time()
        self.right_hand = "closed"
        self.left_hand = "closed"
        # Reset flags
        self.classifier_done = False
        # Clear buffer
        data = clear_lsl_buffer(self.inlet)

    def update_apple(self):
        self.apple_pos[1] += self.apple_speed

        # Reset apple if it goes off the screen
        if self.apple_pos[1] >= SCREEN_HEIGHT+LOAD_BAR_HEIGHT:
            self.failures += 1
            self.reset_for_next_apple()

    def classify(self): 
        if self.game_mode == "define":
            prob = random.choice([self.apple_pos[0] / SCREEN_WIDTH, random.random(), self.apple_pos[0] / SCREEN_WIDTH])
        elif self.game_mode == "training":
            prob = self.apple_pos[0] / SCREEN_WIDTH
        else:
            prob = self.classifier.predict(self.X[-1])
            self.predictions.append(prob)
        return prob
    
    def open_hand(self,prob):
        if prob > 0.5:
            self.right_hand = "open"
        else:
            self.left_hand = "open"

    def check_catch(self):
        if self.apple_pos[1]>=self.player_pos[1]:
            if self.left_hand=="open" and self.apple_pos[0]<SCREEN_WIDTH//2:
                self.score += 1
                self.reset_for_next_apple()
            elif self.right_hand=="open" and self.apple_pos[0]>SCREEN_WIDTH//2:
                self.score += 1
                self.reset_for_next_apple()
    
    def save_data(self):
        if self.game_mode == "training":
            print("Training mode, now computing features after game is complete.")
            self.inverse_operator = create_inverse_operator(self.epochs[0].info)
            for epoch in self.epochs:
                print("\nExtracting features...\n")
                features = extract_features(epoch,self.inverse_operator,-0.1,1.4)
                self.X.append(features)
      
        # Check if the folder corresponding to the subject number exists
        folder = f"data/s{str(self.subject_number).zfill(2)}"
        if not os.path.exists(folder):
            print("New subject, creating new folder")
            os.makedirs(folder)

        # Saving the epochs
        combined_epochs = mne.concatenate_epochs(self.epochs,verbose=False)
        combined_epochs.save(f"{folder}/{time.strftime('%Y-%m-%d_%H%M')}_epo.fif")
        # Saving the features
        np.save(f"{folder}/{time.strftime('%Y-%m-%d_%H%M')}_features.npy", self.X)

        
    def show_menu(self):
        menu_font = pygame.font.SysFont(None, 36)
        title_font = pygame.font.SysFont(None, 54)
        title = title_font.render('Apple Catcher Game', True, (0, 0, 0))
        start_text = menu_font.render('Start Game', True, (0, 0, 0))
        quit_text = menu_font.render('Quit', True, (0, 0, 0))

        title_rect = title.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        start_rect = start_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 100))
        quit_rect = quit_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 150))

        mode_font = pygame.font.SysFont(None, 24)
        mode_0 = mode_font.render('Test', True, (0,0,0))
        mode_1 = mode_font.render('Training', True, (0,0,0))
        mode_2 = mode_font.render('Define', True, (0,0,0))
        mode_0_rect = mode_0.get_rect(center=(SCREEN_WIDTH/4,SCREEN_HEIGHT-100))
        mode_1_rect = mode_1.get_rect(center=(SCREEN_WIDTH/2,SCREEN_HEIGHT-100))
        mode_2_rect = mode_2.get_rect(center=((SCREEN_WIDTH*3)/4,SCREEN_HEIGHT-100))

        # Variables
        COULEUR_CHAMP_TEXTE = (200, 200, 200)
        COULEUR_CHAINE = (255, 255, 255)
        COULEUR_BORDURE = (0, 0, 0)
        # Police pour le texte
        imput_font = pygame.font.Font(None, 24)
        chaine_texte = ''
        actif = False
        # Rectangle du champ de texte
        input_rect = pygame.Rect( 560, SCREEN_HEIGHT-70, 50, 32)

        end_value_txt_font = pygame.font.SysFont(None, 24)
        end_value_txt = end_value_txt_font.render('End_Value: ', True, (0,0,0))
        end_value_txt_rect = end_value_txt.get_rect(center=(((SCREEN_WIDTH*2)/4,SCREEN_HEIGHT-55)))

        while True:
            self.screen.fill(BACKGROUND_COLOR)
            self.screen.blit(self.tree_image, (SCREEN_WIDTH/2-self.tree_image.get_width()/2, 0))
            self.screen.blit(title, title_rect)
            self.screen.blit(start_text, start_rect)
            self.screen.blit(quit_text, quit_rect)

            self.screen.blit(mode_0, mode_0_rect)
            self.screen.blit(mode_1, mode_1_rect)
            self.screen.blit(mode_2, mode_2_rect)

            self.screen.blit(end_value_txt, end_value_txt_rect)

            # Dessiner le champ de texte
            pygame.draw.rect(self.screen, COULEUR_CHAMP_TEXTE, input_rect)
            pygame.draw.rect(self.screen, COULEUR_BORDURE, input_rect, 2)
            # Texte à afficher dans le champ de texte
            txt_surface = imput_font.render(chaine_texte, True, COULEUR_CHAINE)
            # Dessiner le texte
            self.screen.blit(txt_surface, (input_rect.x + 5, input_rect.y + 10))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if start_rect.collidepoint(mouse_pos):
                        return 'start'
                    elif quit_rect.collidepoint(mouse_pos):
                        pygame.quit()
                        sys.exit()
                    if mode_0_rect.collidepoint(mouse_pos):
                        self.game_mode = "test"
                        mode_0 = mode_font.render('Test', True, (255,0,0))
                        mode_1 = mode_font.render('Training', True, (0,0,0))
                        mode_2 = mode_font.render('Define', True, (0,0,0))
                    if mode_1_rect.collidepoint(mouse_pos):
                        self.game_mode = "training"
                        mode_0 = mode_font.render('Test', True, (0,0,0))
                        mode_1 = mode_font.render('Training', True, (255,0,0))
                        mode_2 = mode_font.render('Define', True, (0,0,0))
                    if mode_2_rect.collidepoint(mouse_pos):
                        self.game_mode = "define"
                        mode_0 = mode_font.render('Test', True, (0,0,0))
                        mode_1 = mode_font.render('Training', True, (0,0,0))
                        mode_2 = mode_font.render('Define', True, (255,0,0))

                    if input_rect.collidepoint(event.pos):
                        actif = not actif
                    else:
                        actif = False

                elif event.type == pygame.KEYDOWN:
                    if actif:
                        if event.key == pygame.K_RETURN:
                            self.end_value=int(chaine_texte)
                            chaine_texte = ''
                        elif event.key == pygame.K_BACKSPACE:
                            chaine_texte = chaine_texte[:-1]
                        else:
                            if event.unicode.isdigit():  # Assure que seuls les chiffres sont entrés
                                chaine_texte += event.unicode

            pygame.display.flip()
            pygame.time.Clock().tick(FPS)
        
    
    def run(self):
        self.apple_distribution =[1]*(self.end_value//2) + [0]*(self.end_value//2)
        random.shuffle(self.apple_distribution)
        self.apple_distribution.append(-1) # To increase the length so we dont access out of bounds
        self.apple_pos = self.get_random_starting_position()

        while (self.score + self.failures) < self.end_value:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.elapsed_time = time.time()-self.start_time
        
            # When sufficient time has passed after the marker event, collect data
            if self.elapsed_time > (self.before_marker_time+self.marker_time+0.5) and (self.classifier_done == False):
                sample,timestamps = collect_data(self.inlet, self.inlet_info,self.offset)
                epoch = sample_to_epoch(sample,timestamps,self.inlet_info,self.apple_pos[0])
                self.epochs.append(epoch)

                if self.game_mode =="test":
                    features = extract_features(epoch,self.inverse_operator,-0.1,1.4)
                    self.X.append(features)
                    self.Y.append(epoch.events[0,-1])
                    
                prob = self.classify()
                self.open_hand(prob)
                self.classifier_done = True

            self.check_catch()

            self.update_apple()

            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        if self.game_mode == "test":
            save_results(self.predictions, self.Y,self.subject_number)
            print_results(self.predictions, self.Y)

        self.save_data() 
    
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Apple_catcher_game()
    print("\nGame started!\n")

    if game.show_menu()=='start':
        game.start_time=time.time()
        game.run()