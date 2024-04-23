import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, target_model, lr, gamma, update_target_frequency=1000):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.update_target_frequency = update_target_frequency
        self.steps_since_last_target_update = 0
        self.mse_log = []  # List to store MSE loss values

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).view(-1)

        Q = self.model(states)
        Q_target = Q.clone().detach()

        for idx in range(len(next_states)):
            action_idx = actions[idx].item()
            reward_value = rewards[idx].item() if rewards.dim() > 1 else rewards.item()

            done_value = bool(dones[idx].item()) if isinstance(dones, torch.Tensor) else bool(dones)

            if done_value:
                Q_target[idx] = reward_value
            else:
                next_state = next_states[idx].unsqueeze(0)
                next_Q = self.target_model(next_state)
                max_next_Q = torch.max(next_Q).item()

                if action_idx >= Q_target.shape[1]:
                    continue

                Q_target[idx, action_idx] = reward_value + self.gamma * max_next_Q

        self.optimizer.zero_grad()
        loss = self.criterion(Q, Q_target)
        loss.backward()
        self.optimizer.step()

        self.steps_since_last_target_update += 1
        if self.steps_since_last_target_update >= self.update_target_frequency:
            self.update_target_model()
            self.steps_since_last_target_update = 0

        self.mse_log.append(loss.item())  # Log the MSE loss
        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def calculate_mse(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).view(-1)

        Q = self.model(states)
        Q_target = Q.clone().detach()

        for idx in range(len(next_states)):
            action_idx = actions[idx].item()
            reward_value = rewards[idx].item() if rewards.dim() > 1 else rewards.item()

            done_value = bool(dones[idx].item()) if isinstance(dones, torch.Tensor) else bool(dones)

            if done_value:
                Q_target[idx] = reward_value
            else:
                next_state = next_states[idx].unsqueeze(0)
                next_Q = self.target_model(next_state)
                max_next_Q = torch.max(next_Q).item()

                if action_idx >= Q_target.shape[1]:
                    continue

                Q_target[idx, action_idx] = reward_value + self.gamma * max_next_Q

        mse_loss = F.mse_loss(Q, Q_target)
        return mse_loss.item()
