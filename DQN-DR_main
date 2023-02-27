import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_excel("PHD_THESIS.xlsx")

# Parámetros de DQN
BUFFER_SIZE = 1000  # Tamaño del buffer de experiencia
BATCH_SIZE = 32  # Tamaño del lote de muestras aleatorias tomadas del buffer
GAMMA = 0.9  # Factor de descuento para recompensas futuras
EPSILON_START = 1  # Valor inicial de epsilon (exploración)
EPSILON_END = 0.05  # Valor final de epsilon (explotación)
EPSILON_DECAY = 500  # Número de pasos para reducir epsilon
TARGET_UPDATE = 100  # Frecuencia de actualización del modelo objetivo

# Parámetros del entorno
Learning_rate = 10
Initial_demand = df.Demand.values
Wholesale_price = df.Market_Price.values
Max_limit_InitialPrice = df.Max_price.values
Min_limit_InitialPrice = df.Min_Price.values
Initial_price = df.p0.values


# Red neuronal
class DQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Agente DQN
class DQNAgent:
    def __init__(self):
        self.q_net = DQNet()
        self.target_net = DQNet()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = []
        self.steps_done = 0
        self.epsilon = EPSILON_START

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float()
                q_values = self.q_net(state_tensor)
                action = torch.argmax(q_values).item()
                return action

    def update_replay_buffer(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))
        if len(self.replay_buffer) > BUFFER_SIZE:
            self.replay_buffer.pop(0)

    def optimize_model(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        transitions = random.sample(self.replay_buffer, BATCH_SIZE)
        batch = list(zip(*transitions))
        state_batch = torch.from_numpy(np.stack(batch[0])).float()
        action_batch = torch.tensor(batch[1]).unsqueeze(1)
        next_state_batch = torch.from_numpy(np.stack(batch[2])).float()
        reward_batch = torch.tensor(batch[3]).unsqueeze(1)
        done_batch = torch.logical_not(torch.tensor(batch[4])).unsqueeze(1)

        q_values = self.q_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q_values = reward_batch + GAMMA * next_q_values * done_batch
        loss = nn.functional.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            if self.epsilon > EPSILON_END:
                self.epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY


class Environment:
    def __init__(self, steps):
        self.steps_left = steps
        self.current_step = 0
        self.state = np.array(
            [Initial_demand[self.current_step], Wholesale_price[self.current_step], Initial_price[self.current_step], Max_limit_InitialPrice[self.current_step],
             Min_limit_InitialPrice[self.current_step]])

    def reset(self):
        self.steps_left = len(Initial_demand)
        self.current_step = 0
        self.state = np.array(
            [Initial_demand[self.current_step], Wholesale_price[self.current_step], Initial_price[self.current_step], Max_limit_InitialPrice[self.current_step],
             Min_limit_InitialPrice[self.current_step]])

    def step(self, action):
        self.steps_left -= 1
        self.current_step = min(self.current_step + 1, 23)

        done = False

        if action == 0:  # Reduce price
            if Initial_price[self.current_step] > Min_limit_InitialPrice[self.current_step]:
                Initial_price[self.current_step] -= 0.10 * Initial_price[self.current_step]

        elif action == 1:  # Increase price
            if Initial_price[self.current_step] < Max_limit_InitialPrice[self.current_step]:
                Initial_price[self.current_step] += 0.10 * Initial_price[self.current_step]

        else:  # Stay
            pass

        E = aut_elas(Initial_demand, Initial_price).difference()
        B = -Benefit(Initial_price, Wholesale_price, Initial_price, E) + 0
        comfort = Initial_price[self.current_step] * Initial_demand[self.current_step]
        reward = (comfort) - (1e-3) * abs(Initial_price[self.current_step] - Initial_demand[self.current_step]) - B

        if self.steps_left == 0:
            done = True

        next_state = np.array(
            [Initial_demand[self.current_step], Wholesale_price[self.current_step], Initial_price[self.current_step], Max_limit_InitialPrice[self.current_step],
             Min_limit_InitialPrice[self.current_step]])

        return next_state, reward, done, Initial_price


class aut_elas:
    Dif = np.zeros(24)

    def __init__(self, do, po):
        self.a = np.array(np.diff(do))
        self.b = np.array(np.diff(po))
        self.Dif_n = (self.a / self.b)
        self.Dif = np.transpose(self.Dif)

    def difference(self):
        for index_i in range(23):
            self.Dif[index_i] = self.Dif_n[index_i]
        self.Dif[23] = self.Dif[22]
        Elasticity = np.zeros((24, 24))
        for index_i in range(24):
            for index_j in range(24):
                Elasticity[index_i, index_j] = Initial_price[index_j] / Initial_demand[index_i] * self.Dif[index_i]
        return Elasticity


def Benefit(price, p_w, p_0, elasticity):
    Bias = np.zeros(24)

    for aux_i in range(24):
        d = np.zeros(24)
        for aux_j in range(24):
            elasticity_val = elasticity[aux_i, aux_j]
            d[aux_i] = d[aux_i] + np.log((price[aux_j] / p_0[aux_j]) ** elasticity_val)
        Bias[aux_i] = d[aux_i] * (price[aux_i] - p_w[aux_i])

    out = sum(Bias)

    return out


agent = DQNAgent()
env = Environment(steps=len(Initial_demand))
rewards = []
epsilons = []

for i in range(1000):
    total_reward = 0
    env.reset()
    state = env.state
    while True:
        action = agent.act(state)
        next_state, reward, done, Initial_price = env.step(action)
        agent.update_replay_buffer(state, action, next_state, reward, done)
        agent.optimize_model()
        state = next_state
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)
    epsilons.append(agent.epsilon)

    if i % 100 == 0:
        print(f"Episodio {i}: recompensa total = {total_reward}")

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(rewards)
ax[0].set_xlabel("Episodio")
ax[0].set_ylabel("Recompensa")
ax[1].plot(epsilons)
ax[1].set_xlabel("Paso")
ax[1].set_ylabel("Epsilon")
plt.show()

print(Initial_price)
