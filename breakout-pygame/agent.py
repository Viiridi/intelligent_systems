import torch
import random
import numpy as np
from collections import deque
from game import BreakoutGameAI
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 300_000
BATCH_SIZE = 256
LEARNING_RATE = 0.00005
HEIGHT = 600
WIDTH = 800

GAMMA = 0.95
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

PADDLE_SPEED = 12

class Agent:
    def __init__(self, bricks, epsilon_decay=EPSILON_DECAY):
        input_size = 16
        hidden_size = 1024
        output_size = 3
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.target_model = Linear_QNet(input_size, hidden_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.epsilon_decay = EPSILON_DECAY

        self.n_games = 0
        self.epsilon_min = EPSILON_MIN
        self.epsilon_max = EPSILON_MAX
        self.epsilon = self.epsilon_max
        self.epsilon_decay = epsilon_decay

        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        self.bricks = bricks

        self.trainer = QTrainer(self.model, self.target_model, lr=LEARNING_RATE, gamma=self.gamma)

        # Add a list to store exploration rates
        self.exploration_rates = []
        self.q_value_predictions = []  # List to store predicted Q-values during training steps


    def get_action(self, state, episode_number):
        # Calculate epsilon based on annealing schedule
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (self.epsilon_decay ** episode_number))
        final_move = [0, 0, 0]
        if random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Choose action with the highest Q-value
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state_tensor)
        self.q_value_predictions.append(prediction.detach().numpy())  # Store predicted Q-values


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
            states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in mini_sample])
        else:
            states, actions, rewards, next_states, dones = zip(*self.memory)


        states = np.array([s.numpy() for s in states], dtype=np.float32)
        next_states = np.array([ns.numpy() for ns in next_states], dtype=np.float32)

        states = torch.tensor(states)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(next_states)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def remember(self, state, action, reward, next_state, done):
            experience = (state, action, reward, next_state, done)
            self.memory.append(experience)
            # print("Added experience to memory:", experience)


    def get_state(self, game):
        paddle_x = game.paddle.x
        paddle_width = game.paddle.width
        ball_x = game.ball.x
        ball_y = game.ball.y
        ball_speed_x, ball_speed_y = game.ball_speed
        paddle_speed = game.paddle_speed

        # Calculate distance to closest brick and its relative position
        min_distance_to_brick = float('inf')
        closest_brick_x, closest_brick_y = None, None
        for brick, _ in game.bricks:
            distance_to_brick = np.linalg.norm(np.array([ball_x, ball_y]) - np.array([brick.x + game.brick_width / 2, brick.y + game.brick_height / 2]))
            if distance_to_brick < min_distance_to_brick:
                min_distance_to_brick = distance_to_brick
                closest_brick_x = brick.x
                closest_brick_y = brick.y

        # Calculate relative position of closest brick
        relative_brick_x = (closest_brick_x - ball_x) / game.WIDTH
        relative_brick_y = (closest_brick_y - ball_y) / game.HEIGHT

        # Additional features
        ball_future_x = ball_x + ball_speed_x  # Predicted future position of the ball
        ball_future_y = ball_y + ball_speed_y
        distance_to_wall_left = ball_x / game.WIDTH  # Distance to the left wall
        distance_to_wall_right = (game.WIDTH - ball_x) / game.WIDTH  # Distance to the right wall
        distance_to_ceiling = ball_y / game.HEIGHT  # Distance to the ceiling

        # Calculate paddle and ball velocities
        paddle_velocity = paddle_speed / PADDLE_SPEED  # Normalize paddle velocity
        ball_velocity_x = ball_speed_x / game.WIDTH  # Normalize ball velocity in x-direction

        # Calculate additional features here
        # For example, distance to the nearest edge or center of the screen

        state = [
            paddle_x / game.WIDTH,
            paddle_width / game.WIDTH,
            ball_y / game.HEIGHT,
            ball_x / game.WIDTH,
            ball_speed_x / game.WIDTH,
            ball_speed_y / game.HEIGHT,
            min_distance_to_brick / np.sqrt(game.WIDTH**2 + game.HEIGHT**2),  # Normalized distance to closest brick
            paddle_velocity,
            relative_brick_x,  # Relative position of closest brick (x-coordinate)
            relative_brick_y,  # Relative position of closest brick (y-coordinate)
            ball_future_x / game.WIDTH,  # Predicted future position of the ball (x-coordinate)
            ball_future_y / game.HEIGHT,  # Predicted future position of the ball (y-coordinate)
            distance_to_wall_left,  # Distance to left wall
            distance_to_wall_right,  # Distance to right wall
            distance_to_ceiling,  # Distance to ceiling
            ball_velocity_x,  # Ball velocity in x-direction
            # Add more features here as needed
        ]

        # Normalize each feature to the range [0, 1]
        min_val = min(state)
        max_val = max(state)
        state = [(s - min_val) / (max_val - min_val) for s in state]

        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state)
        state = state.unsqueeze(0)

        return state

    
    def calculate_average_q_value(self):
        if self.q_value_predictions:
            q_values = np.concatenate(self.q_value_predictions, axis=0)
            return np.mean(q_values, axis=0)
        else:
            return None






def train():
    plot_total_reward = []
    total_reward = 0
    plot_scores = []
    plot_mean_scores = []
    plot_median_scores = []
    bricks_remaining = []
    total_score = 0
    record = 0
    losses = []  # List to store losses
    episode_number = 0

    game = BreakoutGameAI()
    agent = Agent(game.bricks)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, episode_number)

        reward, done, score = game.play_step(final_move)
        reward = torch.tensor(reward, dtype=torch.float)
        state_new = agent.get_state(game)

        total_reward += reward

        if game.score > total_score:
            total_score = game.score

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            episode_number += 1
            plot_scores.append(total_score)
            plot_total_reward.append(total_reward)
            mean_score = total_score / (len(plot_scores) + 1)
            median_score = calculate_median(plot_scores)
            plot_mean_scores.append(mean_score)
            plot_median_scores.append(median_score)
            bricks_remaining.append(len(game.bricks))

            game.reset()
            agent.n_games += 1

            # Calculate and log loss to determine if the agent is learning
            loss = agent.trainer.train_step(state_old, final_move, reward, state_new, done)
            losses.append(loss)  # Append loss to the list
            print("Loss:", loss)

            # Train long memory after loss calculation
            agent.train_long_memory()

            if total_score > record:
                record = total_score
                agent.model.save()

            # print('Game', agent.n_games, 'Score', total_score, 'Record:', record)
            total_score = 0

            total_reward = [total_reward]
            plot(plot_scores, plot_mean_scores, plot_median_scores, losses, plot_total_reward)  # Pass losses to the plot function

            # Log the exploration rate here as well
            exploration_rate = agent.epsilon
            agent.exploration_rates.append(exploration_rate)
            # print("Exploration rate:", exploration_rate)

            # print("reward", total_reward)

            total_reward = 0

            if agent.n_games % 10 == 0:
                avg_q_value = agent.calculate_average_q_value()
                if avg_q_value is not None:
                    print(f"Average Q-value: {avg_q_value}")
                print(f"After {agent.n_games} games - Highest Score: {record}, Mean Score: {mean_score}, Median Score: {median_score}, Loss: {loss}, Exploration Rate: {exploration_rate}")



def calculate_median(scores):
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    if n % 2 == 0:
        median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    else:
        median = sorted_scores[n // 2]
    return median



if __name__ == '__main__':
    train()
