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
PADDLE_WIDTH, PADDLE_HEIGHT = 100, 10
BALL_RADIUS = 5
PADDLE_SPEED = 12
BALL_SPEED = 4

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
        self.brick_counter = 0
        self.frame_count = 0

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Breakout Game")

        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.score = 0
        self.paddle = pygame.Rect(WIDTH // 2 - PADDLE_WIDTH // 2, HEIGHT - 20, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
        self.ball_speed = [-1 * BALL_SPEED, -BALL_SPEED]

        self.brick_width = 50
        self.brick_height = 20
        self.bricks = []

        self._place_bricks()  # Corrected method call
        self.frame_iteration = 0

        self.brick_counter = 0

        self.paddle_speed = 0  # Add paddle speed attribute

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
            self.paddle_speed = PADDLE_SPEED  # Update paddle speed
        elif action[2] == 1:  # Move left
            self.paddle.x -= PADDLE_SPEED
            self.paddle_speed = -PADDLE_SPEED  # Update paddle speed
        else:
            self.paddle_speed = 0  # Update paddle speed when not moving

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
        reward = 0  # Initialize reward here
        self.frame_count += 1

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
            reward = 1  # Small positive reward for colliding with the paddle

        if self.ball_speed[1] > 0:  # Ensure the ball is moving downward
            if abs(self.paddle.centerx - self.ball.centerx) < 20:  # Adjust threshold as needed
                reward = 1  # Reward for being close to the ball's x-coordinate    

        # Check if ball falls off the bottom of the screen
        if self.ball.bottom >= HEIGHT:
            # Game over, reset the game
            self.reset()
            game_over = True
            reward = -5  # Negative reward for game over
            return reward, game_over, self.score

        # Check for collision with bricks
        for index, (brick, row) in enumerate(self.bricks):
            if self.ball.colliderect(brick):
                # Remove the brick
                self.bricks.pop(index)
                self.ball_speed[1] = -self.ball_speed[1]

                # Increment the brick counter
                self.brick_counter += 1

                # If it's not the first brick, increase score and reward
                if self.brick_counter > 1:
                    self.score += 100
                    reward = 5

                game_over = len(self.bricks) == 0
                return reward, game_over, self.score

        self._update_ui()

        reward = 0.01

        game_over = False

        self.clock.tick(10000)  # Adjusted frame rate to 120 FPS for faster gameplay
        return reward, game_over, self.score

    def reset_ball(self):
        # Reset the ball's position to the center of the screen
        self.ball.x = WIDTH // 2 - BALL_RADIUS
        self.ball.y = HEIGHT // 2 - BALL_RADIUS
        # Reset the ball's speed
        self.ball_speed = [-1 * BALL_SPEED, -BALL_SPEED]


