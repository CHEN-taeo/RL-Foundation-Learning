
# RL-Foundation-Learning
强化学习地基学习计划

## 学习承诺
未来两周，我只死磕单智能体强化学习（RL）基础，不跳级、不贪多、不被其他方向分心。
每天写代码、做笔记、提交 GitHub，做到真正落地、真正理解。

目标：两周内独立手写 DQN 并让 CartPole 稳定收敛。

## 📁 项目结构
RL-Foundation-Learning/
├── code/
│   ├── marl_demo.py              # MARL演示：多智能体追逐对抗
│   ├── FrozenLake_v1.py          # 环境交互演示：冰湖陷阱
│   ├── Q-table.py                # Q-table学习中的ε-greedy策略实现
│   ├── Q_learning_update.py      # Q_learning贝尔曼更新公式完整实现
│   └── daily_note/
│       ├── day_env_setup_note.md # 环境搭建笔记
│       ├── day1_understanding_of_Five-Tuple.md  # Day1 RL五元组理解
│       ├── Q_table.md            # Q-table核心概念解析
│       └── Q_learning_update.md  # 贝尔曼更新公式原理说明

---

## 📝 文件说明

### `code/marl_demo.py`
**主题：多智能体追逐对抗演示**

使用 PettingZoo 的 MPE 模块中的 `simple_tag_v3` 环境，跑通了 MARL 领域经典的"追逐-逃跑"场景。

- 环境配置：3个追逐者（红）、1个逃跑者（绿）、2个障碍物（黑）
- 策略：随机策略，仅用于演示环境交互流程
- 核心目的：理解多智能体环境的 `agent_iter()` 循环、`env.last()` 观测获取、`termination/truncation` 终止判断

> ⚠️ 当前为可视化演示模式（`render_mode="human"`），正式训练时需关闭渲染。

---

### `code/FrozenLake_v1.py`
**主题：冰湖环境交互演示**

使用 Gymnasium 的 `FrozenLake-v1` 环境，完成了单智能体与环境的基础交互循环。

- 环境配置：`is_slippery=False`（关闭随机滑动，降低初期难度）
- 策略：随机策略（`action_space.sample()`）
- 核心目的：理解 `env.reset()` → `env.step()` → `done` 的完整交互流程，以及五元组在代码中的对应位置

---

### `code/Q-table.py`
**主题：Q-table学习中的ε-greedy策略实现**

实现了强化学习中平衡探索与利用的核心策略，通过动态调整探索概率解决「局部最优」与「经验不足」的矛盾。

- **核心机制**：
  - `ε=1.0`：100%随机探索，适合训练初期积累经验
  - `ε=0.0`：完全贪婪利用，适合训练后期执行最优策略
  - `0<ε<1`：混合策略（如ε=0.3表示30%探索概率）
- **函数特性**：
  - 动态获取动作空间大小（`q_table.shape[1]`）
  - 自动处理多动作Q值相同的情况（返回首个最大值索引）
  - 完整的ε衰减测试脚本，可视化不同探索概率下的行为模式

> 💡 **关键洞察**：该策略是智能体从「盲目试错」到「经验决策」的桥梁，通过ε的递减实现学习过程的平滑过渡。

---

### `code/Q_learning_update.py`
**主题：Q_learning贝尔曼更新公式完整实现**

实现了Q-learning算法的核心更新机制，将理论公式转化为可执行代码，完成智能体经验更新的闭环。

- **贝尔曼方程实现**：
  ```python
  Q[s,a] += α * (r + γ * max(Q[s_next]) - Q[s,a])
  ```
- **参数物理意义**：
  - `α (alpha)`：学习率，控制新旧经验的权重（0.1表示10%信任新经验）
  - `γ (gamma)`：折扣因子，平衡即时奖励与未来回报（0.9表示重视长期收益）
  - `max(Q[s_next])`：预测下一状态的最大潜在价值
- **更新逻辑**：
  1. 计算时序差分误差（TD error）：`r + γ*max(Q[s_next]) - Q[s,a]`
  2. 用学习率缩放误差，避免单次更新的偶然性
  3. 累加到当前Q值，逐步逼近真实价值

> ⚙️ **设计亮点**：分离策略选择（ε-greedy）与价值更新（贝尔曼方程）两个核心模块，符合强化学习算法的标准架构。

---

### `code/daily_note/day_env_setup_note.md`
**环境搭建笔记**

记录了 RL 专属虚拟环境的搭建结果：

| 项目 | 版本 |
|------|------|
| Python | 3.13 |
| PyTorch | 2.2.0 |
| Gymnasium | ✅ |
| PettingZoo | ✅ |
| TensorBoard | ✅ |

---

### `code/daily_note/day1_understanding_of_Five-Tuple.md`
**Day1：RL核心五元组通俗理解**

用 Arduino 避障小车场景，把五元组拆成了能摸得到的东西：

| 五元组 | 对应行为 |
|--------|----------|
| 观测/状态 | 小车识别前方障碍物距离、当前速度 |
| 动作 | 刹车 |
| 奖励 | 未撞到 +1，撞到 -10 |
| 策略 | 判断当前速度：能转弯→转弯，否则→刹车 |
| 回报 | 累计奖励总和，用于评估策略好坏 |

---

### `code/daily_note/Q_table.md`
**Q-table核心概念通俗解析**

| 概念 | 技术定义 | 现实类比 |
|------|----------|----------|
| **Q表本质** | 状态-动作价值矩阵 | 一本记录「在什么位置做什么动作能获得多少奖励」的经验手册 |
| **初始化** | `np.zeros([5,3])` | 全新空白手册，没有任何先验经验 |
| **行/列意义** | 行=状态空间，列=动作空间 | 行=所有可能的路况，列=所有可能的驾驶操作 |
| **更新目标** | 收敛到最优Q值 | 手册记录越来越准确，最终成为「完美驾驶指南」 |

---

### `code/daily_note/Q_learning_update.md`
**贝尔曼更新公式原理精解**

```math
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'}Q(s',a') - Q(s,a) \right]
```

**五步拆解**：
1. **预测未来**：`max(Q[s_next])` - 估计下一状态能获得的最佳累积奖励
2. **折现处理**：`γ * max(Q[s_next])` - 用折扣因子γ降低远期奖励权重（「一鸟在手胜过双鸟在林」）
3. **计算总收益**：`r + γ*max(Q[s_next])` - 即时奖励+折现后的未来奖励
4. **对比误差**：`总收益 - Q[s,a]` - 新估计值与旧记录的差值（TD误差）
5. **渐进更新**：`α * 误差` - 用学习率控制更新步长，避免剧烈波动

> 🌟 **本质**：这不是简单的数值更新，而是智能体通过「预测-验证-修正」的闭环，持续优化其决策认知的过程。

---

## 🗓️ 更新记录

| 日期 | 内容 |
|------|------|
| 2026.04.13 | 新增 `marl_demo.py`，跑通 PettingZoo MPE 追逐环境 |
| 2026.04.14 | 新增 `FrozenLake_v1.py`，完成 Gymnasium 基础交互循环 |
| 2026.04.14 | 归档环境搭建笔记与 Day1 五元组理解 |
| 2026.04.15 | 新增 `Q-table.py`，实现ε-greedy策略及完整测试脚本 |
| 2026.04.16 | 新增 `Q_learning_update.py`，完成贝尔曼更新公式代码落地 |
| 2026.04.16 | 归档Q-learning核心概念解析笔记，建立理论-代码映射 |

---

### 🔧 集成建议
1. **训练流程**：将`epsilon_greedy_policy`作为动作选择器，`Q_learning_update`作为学习器，构建完整Q-learning训练循环
2. **参数调优**：从`α=0.1, γ=0.9, ε=0.3`开始，根据环境复杂度调整（复杂环境可提高ε初始值）
3. **可视化扩展**：建议添加Q表变化热力图，直观观察学习过程（可使用matplotlib.pyplot.imshow）


---
# 单智能体Q-Learning循环训练项目
## 一句话简介
#### 【冰糕】在一次次摔进冰洞的失败中总结经验，最终找到最优路线，囤够食物度过冬天。

---

## 目录 
[课前准备](#课前准备) 
[快速开始](#快速开始) 
[课前引导](#课前引导) 
[课程讲解](#课程讲解) 
[核心代码逻辑讲解](#核心代码逻辑讲解) 
[调参实验的核心结论](#调参实验的核心结论) 
[结果展示](#结果展示)
 [作者信息与后续计划](#作者信息与后续计划)
--- 
### 课前准备
1.运行项目需要的Python环境(python 3.13,亲测有效)。
2.安装所有的第三方库：numpy，gymnasium，matplotlib。
```
pip install numpy gymnasium matplotlib
```
3.为避免和其他项目的库冲突，可以使用虚拟环境。
```python
# 创建虚拟环境 
python -m venv q_learning_env 
# Windows系统激活虚拟环境 q_learning_env\Scripts\activate 
# Mac/Linux系统激活虚拟环境 source q_learning_env/bin/activate
```
### 快速开始
1. 克隆 / 下载本项目到本地
2. 安装所有依赖库（复制粘贴课前准备里的命令）
3. 运行 `python q_learning_loop.py`，即可看到训练过程的打印输出和最终的 reward 曲线图
--- 
### 课前引导
##### 故事导入
> 你见过一个完全不会认路的小人，靠摔1000次冰洞，最后找到最短的路径，避开所有的冰洞吗？
> 我们把这个小人叫做【冰糕】，他和你未来要做的机器人一样，都是在未知的环境中，靠一次次试错，找到最优路径。
##### [[学习]]原因
当我们了解到小人是如何像人一样学会生活时，那我们就可以用同样的方法设计我们的机器人，让他能够不断[[学习]]了解世界，能够处理遇见的问题。
项目原因：设计机器人的[[学习]]模式，制作一个可以自主[[学习]]的机器人
##### 作用价值
- 入门强化[[学习]]的练手项目
- 理解控制变量法的实验模板
- 机器人导航算法的基础原型

---
## 课程讲解
### 第一个问题：核心代码逻辑讲解 
#### 先搞懂：Q表是什么？ 
Q表就是【冰糕】脑子里的「冰湖地图经验之书」——每一行对应冰湖的一个格子（状态），每一列对应一个动作（上下左右），格子里的数字就是【冰糕】对「走这个动作能拿到多少奖励」的预估。 
#### 最核心的一行：贝尔曼更新Q表 这一行代码，就是【冰糕】摔了冰洞/拿到食物后，更新自己「经验之书」的过程，我们一步步拆解： 
```
python Q_table[state, action] += alpha * ( reward + gamma * np.max(Q_table[next_state, :]) - Q_table[state, action] )
```
#### 代码讲解
核心代码：贝尔曼更新Q表——【冰糕】在做出下一个动作之前会对下一个动作进行评估得到值np.max（Q_state[s_next,:])，这个值包含对未来获得奖励的最大累积值。但是预测的数值由于未来的不确定，所以会乘上gamma值打折，为了估值更加准确，会把打折后的值减掉上一次的对该状态的估值得到两者之间的差值：np.max(Q_table[s_next,:])-Q_table[state].然后再加上一个此时的奖励值得到较为准确的预测值。最后乘上alpha（学习率）来更新自己的Q表【经验之书】。
代码(有详细注释）：

```
# 主题：单智能体循环训练  
# 时间：2026-04-17  
# 作者：陈韬  
'''  
【基准对照组】  
alpha=0.1  
gamma=0.9  
epsilon=0.1（固定值，无衰减）  
max_episodes=1000  
max_step_per_episode=100  
环境：FrozenLake-v1, is_slippery=False  
'''  
  
# 1.  导入必要的库（比如gym、numpy、tensorboard的SummaryWriter）  
import numpy as np  
import gymnasium as gym  
# from tensorboard import SummaryWriter  
import matplotlib.pyplot as plt  
import random # 固定所有随机数种子，保证结果可以复现  
random.seed(42)# 固定Python内置随机数种子  
np.random.seed(42)# 固定numpy随机数种子  
# 固定gymnasium环境的随机数种子（在env.reset()和env.step()里生效）  
# 3.  初始化环境（比如我们的小车巡检栅格环境）  
env = gym.make("FrozenLake-v1",is_slippery=False,render_mode="")  
# 假设我们有一个自定义的环境叫Frozenlake-v1  
env.action_space.seed(42)  
env.observation_space.seed(42)  
# 2.  初始化超参数：alpha=0.1, gamma=0.9, epsilon=0.1, max_episodes=1000, max_steps_per_episode=100  
alpha=0.1  
gamma=0.9  
epsilon=0.1  
max_episodes=1000  
max_step_per_episode=100  
  
# 4.  初始化Q表：全0，行数=状态数，列数=动作数  
Q_table = np.zeros([env.observation_space.n,env.action_space.n])  
# 5.  初始化all_rewards列表，用来存每个episode的总reward  
all_rewards=[]  
# 在超参数下面加一行（第20行左右）  
# checkpoint_states = [10, 20, 30]  # 假设的巡检点状态  
epsilon_decay = 0.995  # 每次训练后，epsilon乘以0.995，慢慢变小  
min_epsilon = 0.01  # epsilon最小降到0.01，保留一点探索  
# 6.  初始化TensorBoard的SummaryWriter  
# writer = SummaryWriter()    # 原来：writer = SummaryWriter;Summarywriter相当于是图纸，你不能用图纸来画画，所以在SummaryWriter的后面要加上一个括号（）  
# 7.  开始外层循环：  
for episode in range(max_episodes):  
    # 7.1 重置环境：state = 环境.reset()  
    state,info=env.reset()  # 原来：state=env.reset();gymnasium的reset会返回两个东西，就像是点了一杯奶茶，你最后只拿了奶茶，没有拿吸管，所以还需要一个info接住其他信息  
    # 7.2 重置临时变量：total_reward = 0, step_count = 0  
    total_reward = 0  
    step_count = 0  
    # 7.3 重置巡检点访问标记：  
   # visited_checkpoints = [False, False, False]  
  
  
    # 7.4 开始内层循环：  
    for step in range(max_step_per_episode):  
        # 生成随机数  
        rand_num = np.random.random()  
        # 7.4.1 用ε-greedy策略选动作：  
        if rand_num < epsilon:  
            # 随机选动作  
            action = np.random.choice(env.action_space.n)  
        else:  
            # 选Q表中当前state对应的最大Q值的动作  
            action = np.argmax(Q_table[state,:])  
        # 7.4.2 执行动作：next_state, reward, done, info = 环境.step(action)  
        next_state,reward,terminated,truncated,info = env.step(action)  
            # 去掉done，调整顺序  
            # 7.4.3 【关键】按优先级调整reward：  
        #idx = checkpoint_states.index(next_state)  
        if terminated :  
            if reward == 1:  
                reward = 50  
            else:  
                reward = -10  
  
        #elif visited_checkpoints[idx]==False:  
            reward += 5  
            #visited_checkpoints[idx] = True  
        # 每步基础惩罚  
  
        reward -= 0.1  
            # 7.4.4 【核心】贝尔曼更新Q表：  
        Q_table[state, action] += alpha * ( reward + gamma * np.max(Q_table[next_state, :]) - Q_table[state, action] )  
            # 修改：把max改为np.max  
            # 7.4.5 更新状态：  
        state = next_state  
            # 7.4.6 累加总reward：  
        total_reward+= reward  
            # 修改后：直接累加，不用env.append  
            # 7.4.7 累加步数：  
        step_count += 1  
            # 7.4.8 判断是否终止：  
        if terminated or truncated:  
            break  
    # 7.5 把当前episode的total_reward加入all_rewards列表  
    all_rewards.append(total_reward)  
    # 让epsilon慢慢衰减  
    epsilon = max(min_epsilon, epsilon * epsilon_decay)  
    # 7.6 【记录】每100个episode算平均并记录：  
    if episode % 100 == 0:  
        avg_reward =np.mean(all_rewards[-100:])  
        print("Episode {}, Average Reward: {:.2f}".format (episode,avg_reward))  
        # writer.add_scalar("Average Reward/100 Episodes", avg_reward, episode)  
# 8.  训练结束，关闭TensorBoard的writer  
# writer.close()  
# 9.  保存Q表，保存all_rewards的曲线截图到GitHub  
# 画reward曲线  
plt.figure(figsize=(10, 5))  
plt.plot(all_rewards, label='Episode Reward', alpha=0.5)  # alpha=0.5让曲线半透明，不那么乱  
# 画每100个episode的平均reward曲线（更平滑）  
avg_rewards = []  
window_size = 100  
for i in range(len(all_rewards) - window_size + 1):  
    avg_rewards.append(np.mean(all_rewards[i:i+window_size]))  
plt.plot(range(window_size, len(all_rewards)+1), avg_rewards, label=f'Average Reward ({window_size} Episodes)', linewidth=2, color='red')  
plt.xlabel('Episode')  
plt.ylabel('Total Reward')  
plt.title('Q-Learning Training on FrozenLake-v1')  
plt.legend()  
plt.grid(True)  
# 保存图片  
plt.savefig('q_learning_frozenlake.png')  
plt.show()
```
### 结果展示
![[屏幕截图 2026-04-18 091713.png]]

![[屏幕截图 2026-04-18 195252.png]]
### 第二个问题：调参实验的核心结论
```
'''  
【基准对照组】  
alpha=0.1  
gamma=0.9  
epsilon=0.1（固定值，无衰减）  
max_episodes=1000  
max_step_per_episode=100  
环境：FrozenLake-v1, is_slippery=False  
'''  
```
---
##### 【实验组 1：学习率 alpha 的影响实验（唯一变量：alpha）】

- 固定所有参数和基准组完全一致，仅修改 alpha
- 测试值：0.01、0.05、0.2、0.5（覆盖从慢学到快学的全区间）
- 核心测试目标：验证 alpha 对收敛速度、最终性能、稳定性的影响
- 核心结论：alpha 越大，收敛速度越快，但后期波动越大；alpha 越小，收敛越慢，但最终结果越稳定

---

##### 【实验组 2：折扣因子 gamma 的影响实验（唯一变量：gamma）】

- 固定所有参数和基准组完全一致，仅修改 gamma
- 测试值：0.5、0.8、0.95、0.99（覆盖从短视到远见的全区间）
- 核心测试目标：验证 gamma 对智能体决策远见性的影响
- 核心结论：gamma 越接近 1，智能体越看重未来的最终奖励，越愿意走长路径避障，最终 reward 上限越高；gamma 越接近 0，智能体越短视，容易陷入局部最优

---

##### 【实验组 3：探索率 epsilon 策略的影响实验（唯一变量：epsilon 策略）】

- 固定所有参数和基准组完全一致，仅修改 epsilon 的策略
- 3 组对比策略：
    
    1. 固定 epsilon=0.01（极低探索，纯利用）
    2. 固定 epsilon=0.2（高频探索，纯探索）
    3. epsilon 衰减策略（初始 0.1，每轮乘以 0.995，最低 0.01）
    
- 核心测试目标：验证探索 - 利用平衡对训练效果的影响
- 核心结论：epsilon 衰减策略收敛最快、最终 reward 最高、波动最小；固定 0.01 容易陷入局部最优，根本学不会；固定 0.2 波动极大，收敛极慢。

> 记住：Q-learning的精髓不在公式本身，而在于「用误差驱动认知进化」的思想。这两个文件共同构成了价值迭代的基础模块，后续可扩展为DQN等深度强化学习算法的核心组件。
```
