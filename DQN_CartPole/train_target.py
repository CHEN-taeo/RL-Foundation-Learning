# 主题：深度 Q 网络（Deep Q-Network，简称 DQN）算法的核心训练函数（train ()）完整实现
# 时间：2026-04-23——2026-04-24
# 作者：陈韬
import torch
# torch库常用于准确数字计算。
# 用改作业类比，你是学生，网络是你的“答题本”。target_q是“标准答案”

import torch

def target_network(target_net, next_states):
    return target_net(next_states)

def train(network, target_net, buffer, optimiser, batch_size=32, gamma=0.99):
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = network(states)
    actions_unsqueezed = actions.unsqueeze(1)
    current_q = q_values.gather(1, actions_unsqueezed)

    with torch.no_grad():
        next_q_values = target_network(target_net, next_states)
        q_max_values = next_q_values.max(dim=1)[0]
        q_target = rewards + gamma * q_max_values * (1 - dones)

    q_target_unsqueezed = q_target.unsqueeze(1)

    criterion = torch.nn.MSELoss()
    loss = criterion(current_q, q_target_unsqueezed)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return loss.item()



# ---------------- 极简测试框架（仅用于测试打印loss） ----------------

