import torch
import random
import numpy as np
from collections import deque
from game import BreakoutGameAI
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
HEIGHT = 600
WIDTH = 800
GAMMA = 0.99
EPSILON_MIN = 0.02
EPSILON_MAX = 1.0
EPSILON_DECAY = 0.998
PADDLE_SPEED = 12

class Agent:
    def __init__(self, bricks, initial_epsilon=EPSILON_MAX, final_epsilon=EPSILON_MIN, exploration_anneal_episodes=1000):
        input_size = 6  # Adjust input size
        hidden_size = 512  # Adjust hidden layer size
        output_size = 3  # Adjust output size
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.target_model = Linear_QNet(input_size, hidden_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.epsilon_min = final_epsilon
        self.epsilon_max = initial_epsilon
        self.epsilon_decay = EPSILON_DECAY
        self.exploration_anneal_episodes = exploration_anneal_episodes
        self.epsilon = self.epsilon_max

        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        self.bricks = bricks

        self.trainer = QTrainer(self.model, self.target_model, lr=LEARNING_RATE, gamma=self.gamma)

        self.exploration_rates = []  # List to store exploration rates
        self.q_value_predictions = []  # List to store predicted Q-values during training steps
        
        # Initialize n_games attribute
        self.n_games = 0

    def get_action(self, state, episode_number):
        # Anneal epsilon linearly over a specified number of episodes
        if episode_number < self.exploration_anneal_episodes:
            self.epsilon = self.epsilon_max - (self.epsilon_max - self.epsilon_min) * (episode_number / self.exploration_anneal_episodes)
        else:
            self.epsilon = self.epsilon_min
        
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

    def get_state(self, game):
        ball_x = game.ball.x
        ball_y = game.ball.y
        ball_speed_x, ball_speed_y = game.ball_speed
        paddle_x = game.paddle.x
        remaining_bricks = [brick_pos for brick_pos, _ in game.bricks]  # Extract brick positions
        time = game.frame_count

        # Flatten the list of brick positions
        flattened_bricks = [pos for brick_pos in remaining_bricks for pos in brick_pos]

        state = [
            ball_x / game.WIDTH,
            ball_y / game.HEIGHT,
            ball_speed_x / game.WIDTH,
            ball_speed_y / game.HEIGHT,
            paddle_x / game.WIDTH,
            time
        ]

        # Convert all elements to floats
        state = [float(s) for s in state]

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
