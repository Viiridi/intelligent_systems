import torch
import random
import numpy as np
from collections import deque
from game import BreakoutGameAI
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.0001
HEIGHT = 600
WIDTH = 800

class Agent:
    def __init__(self, bricks):
        # Initialize main and target networks
        input_size = 7
        hidden_size = 256
        output_size = 3
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.target_model = Linear_QNet(input_size, hidden_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Set target model to evaluation mode

        self.n_games = 0
        self.epsilon_min = 0.1
        self.epsilon_max = 1.0
        self.epsilon = self.epsilon_max
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)

        # Store bricks as an attribute
        self.bricks = bricks

        # Initialize QTrainer with both main and target models
        self.trainer = QTrainer(self.model, self.target_model, lr=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        paddle_x = game.paddle.x
        paddle_width = game.paddle.width
        ball_x = game.ball.x
        ball_y = game.ball.y
        ball_speed_x, ball_speed_y = game.ball_speed

        # Calculate distances from the ball to the paddle and bricks
        distance_to_paddle = abs(ball_y - game.paddle.y)

        min_distance_to_brick = float('inf')
        for brick, _ in game.bricks:
            # Calculate distance to each brick and find the minimum
            distance_to_brick = np.linalg.norm(np.array([ball_x, ball_y]) - np.array([brick.x + game.brick_width / 2, brick.y + game.brick_height / 2]))
            min_distance_to_brick = min(min_distance_to_brick, distance_to_brick)

        # Normalize distances
        normalized_distance_to_paddle = distance_to_paddle / HEIGHT
        normalized_distance_to_brick = min_distance_to_brick / np.sqrt(HEIGHT**2 + WIDTH**2)  # Diagonal length of the game window

        # Construct the state representation
        state = [
            paddle_x / game.WIDTH,
            normalized_distance_to_paddle,
            paddle_width / game.WIDTH,
            ball_y / game.HEIGHT,
            ball_x / game.WIDTH,
            ball_speed_x / game.WIDTH,
            ball_speed_y / game.HEIGHT,
            normalized_distance_to_brick
        ]

        # Convert state to a single numpy array
        state = np.array(state, dtype=np.float32)

        # Convert the numpy array to a tensor
        state = torch.tensor(state)

        # Ensure state has the correct shape (1x7)
        state = state.unsqueeze(0)  # Add a batch dimension

        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = np.array([s.numpy() for s in states], dtype=np.float32)
        next_states = np.array([ns.numpy() for ns in next_states], dtype=np.float32)

        states = torch.tensor(states)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(next_states)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)  # Reshape actions tensor

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)  # Decay epsilon more gradually

        final_move = [0, 0, 0]

        if random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_median_scores = []  # Initialize list for median scores
    bricks_remaining = []  # Track the number of remaining bricks
    total_score = 0
    record = 0

    game = BreakoutGameAI()
    agent = Agent(game.bricks)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)  # Update to receive reward, game over flag, and score
        reward = torch.tensor(reward, dtype=torch.float)  # Convert reward to a tensor
        state_new = agent.get_state(game)

        # Update the score based on the number of bricks destroyed
        if game.score > total_score:
            total_score = game.score

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Store the score before resetting the game
            plot_scores.append(total_score)
            mean_score = total_score / (len(plot_scores) + 1)  # Update mean score
            median_score = calculate_median(plot_scores)  # Calculate median score
            plot_mean_scores.append(mean_score)
            plot_median_scores.append(median_score)  # Append median score
            bricks_remaining.append(len(game.bricks))

            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if total_score > record:
                record = total_score
                agent.model.save()  # Save the model when a new high score is achieved

            print('Game', agent.n_games, 'Score', total_score, 'Record:', record)
            total_score = 0  # Reset total score

            plot(plot_scores, plot_mean_scores, plot_median_scores)  # Pass median scores to plot function
            state_old = state_old.unsqueeze(0)
            agent.train_long_memory()


def calculate_median(scores):
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    if n % 2 == 0:
        # If even number of scores, average the two middle values
        median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    else:
        # If odd number of scores, take the middle value
        median = sorted_scores[n // 2]
    return median            


if __name__ == '__main__':
    train()
