import pygame
import sys
import random
import time
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

font = pygame.font.Font(None, 36)

WIDTH, HEIGHT = 600, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 50, 10
BALL_RADIUS = 5
PADDLE_SPEED = 8
BALL_SPEED = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (191, 64, 191)

brick_color = [RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE]

class Direction(Enum):
    STILL = 1
    RIGHT = 2
    LEFT = 3

class BreakoutGameAI:
    def __init__(self):
        self.WIDTH = 600
        self.HEIGHT = 600
        
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Breakout Game")

        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.score = 0
        self.paddle = pygame.Rect(WIDTH // 2 - PADDLE_WIDTH // 2, HEIGHT - 20, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
        self.ball_speed = [random.choice([-1, 1]) * BALL_SPEED, -BALL_SPEED]

        self.brick_width = 50
        self.brick_height = 20
        self.bricks = []

        self._place_bricks()  # Corrected method call
        self.frame_iteration = 0

    def _update_ui(self):
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, WHITE, self.paddle)
        pygame.draw.ellipse(self.screen, WHITE, self.ball)

        for brick, row in self.bricks:
            pygame.draw.rect(self.screen, brick_color[row], brick)
       
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def _move(self, action):
        # Action mapping: [stay still, move right, move left]
        if action[1] == 1:  # Move right
            self.paddle.x += PADDLE_SPEED
        elif action[2] == 1:  # Move left
            self.paddle.x -= PADDLE_SPEED

        # Ensure the paddle stays within the game window
        self.paddle.x = max(0, min(WIDTH - PADDLE_WIDTH, self.paddle.x))

        self.ball.x += self.ball_speed[0]
        self.ball.y += self.ball_speed[1]

    def is_collision(self, obj1, obj2):
        return obj1.colliderect(obj2)

    def _place_bricks(self):  # Properly defined _place_bricks method
        for row in range(6):
            for col in range(WIDTH // (self.brick_width + 3)):
                brick = pygame.Rect(col * (self.brick_width + 3) + 10, row * (self.brick_height + 3) + 50, self.brick_width, self.brick_height)
                self.bricks.append((brick, row))

    def play_step(self, action):
        self.frame_iteration += 1
        reward = 0  # Initialize reward here

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self._move(action)

        # Check for collisions with walls
        if self.ball.left <= 0 or self.ball.right >= WIDTH:
            self.ball_speed[0] = -self.ball_speed[0]
        if self.ball.top <= 0:
            self.ball_speed[1] = -self.ball_speed[1]

        # Check for collision with paddle
        if self.is_collision(self.ball, self.paddle):
            self.ball_speed[1] = -self.ball_speed[1]
            reward += 0.1  # Small positive reward for colliding with the paddle

        # Check if ball falls off the bottom of the screen
        if self.ball.bottom >= HEIGHT:
            # Game over, reset the game
            self.reset()
            game_over = True
            reward = -10  # Negative reward for game over
            return reward, game_over, self.score

        # Check for collision with bricks
        for brick, row in self.bricks:
            if self.ball.colliderect(brick):
                self.bricks.remove((brick, row))
                self.ball_speed[1] = -self.ball_speed[1]

                # Increase score by 100 for each brick broken
                self.score += 100  

                if len(self.bricks) != 0:
                    reward += 1  # Increment reward for breaking a brick
                else:
                    reward += 10  # Increment reward for completing the game
                    game_over = True
                    return reward, game_over, self.score

        self._update_ui()

        game_over = False

        self.clock.tick(60)
        return reward, game_over, self.score


    def reset_ball(self):
        # Reset the ball's position to the center of the screen
        self.ball.x = WIDTH // 2 - BALL_RADIUS
        self.ball.y = HEIGHT // 2 - BALL_RADIUS
        # Reset the ball's speed
        self.ball_speed = [random.choice([-1, 1]) * BALL_SPEED, -BALL_SPEED]