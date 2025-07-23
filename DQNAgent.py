import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import cv2


class DQNCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.0  # normalize
        conv_out = self.conv(x).reshape(x.size(0), -1)
        return self.fc(conv_out)


class DQNAgent:
    def __init__(self, input_shape, num_actions, lr=1e-4, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 memory_size=50000, batch_size=32, device='cpu'):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device

        self.model = DQNCNN(input_shape, num_actions).to(device)
        self.target_model = DQNCNN(input_shape, num_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action = int(torch.argmax(q_values))

        # Hiển thị ảnh quan sát hiện tại và hành động đã chọn
        img = state.squeeze()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        text = f"Action: {action}"
        cv2.putText(img_bgr, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Agent View", img_bgr)
        cv2.waitKey(1)

        return action

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
