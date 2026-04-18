# 🧊 Q-Learning FrozenLake | 冰糕找食物

> 你见过一个完全不会认路的小人，踩着薄薄的冰面，摔进冰洞一千次，爬出来一千次，最后靠着一身淤青摸出那条最短的路、绕开所有的冰洞吗？
>
> 冬天快来了，他必须找到食物，没有退路。
>
> 我们叫他**【冰糕】**。他跟你以后要做的机器人一模一样——没有地图，没有人教，只能自己在未知的冰湖上一步一步试，摔了记住，爬起来接着走，直到走出那条最优的路。

---

## 📌 一句话简介

**【冰糕】** 在一次次摔进冰洞的失败中总结经验，最终找到最优路线，囤够食物度过冬天。

这是一个基于 **Q-Learning** 算法、在 `FrozenLake-v1` 环境中训练单智能体的完整项目，包含训练循环、调参实验与结果可视化。

---

## 🛠️ 环境准备

**Python 版本：3.13（亲测有效）**

安装依赖：

```bash
pip install numpy gymnasium matplotlib
```

建议使用虚拟环境，避免和其他项目的库冲突：

```bash
# 创建虚拟环境
python -m venv q_learning_env

# Windows 激活
q_learning_env\Scripts\activate

# Mac/Linux 激活
source q_learning_env/bin/activate
```

---

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/CHEN-taeo/RL-Foundation-Learning.git

# 2. 进入目录
cd RL-Foundation-Learning

# 3. 安装依赖
pip install numpy gymnasium matplotlib

# 4. 运行训练
python q_learning_loop.py
```

运行后你会看到训练过程的打印输出，以及最终的 reward 曲线图。

---

## 🧠 核心原理

### Q 表是什么？

Q 表就是**冰糕脑子里的「冰湖地图经验之书」**——

- 每一行对应冰湖的一个格子（状态）
- 每一列对应一个动作（上下左右）
- 格子里的数字是冰糕对「走这步能拿到多少奖励」的预估

一开始全是 0，什么都不知道；摔了一千次之后，这本书就变成了最优路线图。

### 贝尔曼更新：冰糕怎么从失败里学习？

```python
Q_table[state, action] += alpha * (
    reward + gamma * np.max(Q_table[next_state, :]) - Q_table[state, action]
)
```

五步拆解：

| 步骤 | 公式片段 | 白话解释 |
|------|----------|----------|
| ① 预测未来 | `np.max(Q_table[next_state, :])` | 下一步最多能拿多少奖励 |
| ② 折现处理 | `gamma * ...` | 未来的奖励打个折，不如现在的确定（一鸟在手）|
| ③ 算总收益 | `reward + gamma * ...` | 现在的奖励 + 打折后的未来奖励 |
| ④ 算误差 | `总收益 - Q_table[state, action]` | 新估计 vs 旧记录，差了多少 |
| ⑤ 渐进更新 | `alpha * 误差` | 用学习率控制更新幅度，别一次改太猛 |

> **本质**：冰糕不是靠运气找到路的，是靠「预测 → 验证 → 修正」的闭环，把经验一点点写进 Q 表里。

---

## ⚙️ 超参数说明

```python
alpha   = 0.1    # 学习率：每次更新信任多少新经验（10%）
gamma   = 0.9    # 折扣因子：有多看重未来的奖励
epsilon = 0.1    # 探索率：以多大概率随机试新动作
max_episodes          = 1000  # 总训练轮数
max_step_per_episode  = 100   # 每轮最多走多少步
```

ε-greedy 策略：

- `rand < epsilon` → 随机探索（试新路）
- `rand >= epsilon` → 贪心利用（走 Q 表里最好的路）
- `epsilon` 随训练轮数衰减（`× 0.995`，最低降到 `0.01`）

---

## 🔬 调参实验结论

### 实验 1：学习率 alpha 的影响

测试值：`0.01 / 0.05 / 0.1 / 0.2 / 0.5`

| alpha | 收敛速度 | 后期稳定性 |
|-------|----------|------------|
| 小（0.01） | 慢 | 稳 |
| 大（0.5） | 快 | 抖 |

**结论：alpha 越大收敛越快，但后期波动越大；alpha 越小越稳，但要等更久。**

---

### 实验 2：折扣因子 gamma 的影响

测试值：`0.5 / 0.8 / 0.9 / 0.95 / 0.99`

| gamma | 智能体性格 | 表现 |
|-------|------------|------|
| 接近 0 | 短视，只顾眼前 | 容易陷入局部最优 |
| 接近 1 | 有远见，愿意走长路绕障碍 | reward 上限更高 |

**结论：gamma 越接近 1，冰糕越愿意多走几步绕开冰洞，最终路线质量越高。**

---

### 实验 3：探索策略 epsilon 的影响

三种策略对比：

| 策略 | 描述 | 结果 |
|------|------|------|
| 固定 0.01 | 几乎不探索，只走熟悉的路 | 容易卡在局部最优，学不好 |
| 固定 0.2 | 一直乱试 | 收敛极慢，波动极大 |
| **衰减策略** | 从 0.1 开始，慢慢降到 0.01 | **收敛最快、reward 最高、最稳定** |

**结论：先大胆探索，再稳定利用——衰减策略完胜。**

---

## 📊 训练结果

训练 1000 轮后的 reward 曲线：

![训练结果](q_learning_frozenlake.png)

红线为每 100 个 episode 的滑动平均，可以清晰看到冰糕从「乱撞」到「找到最优路线」的学习过程。

---

## 📁 项目结构

```
RL-Foundation-Learning/
├── q_learning_loop.py          # 主训练脚本（带完整注释）
├── q_learning_frozenlake.png   # 训练结果曲线图
├── code/
│   ├── Q-table.py              # ε-greedy 策略实现
│   ├── Q_learning_update.py    # 贝尔曼更新公式实现
│   ├── FrozenLake_v1.py        # Gymnasium 基础交互演示
│   ├── marl_demo.py            # PettingZoo 多智能体演示
│   └── daily_note/
│       ├── day1_understanding_of_Five-Tuple.md
│       ├── Q_table.md
│       └── Q_learning_update.md
└── README.md
```

---

## 🗓️ 更新记录

| 日期 | 内容 |
|------|------|
| 2026.04.13 | 新增 `marl_demo.py`，跑通 PettingZoo MPE 追逐环境 |
| 2026.04.14 | 新增 `FrozenLake_v1.py`，完成 Gymnasium 基础交互循环 |
| 2026.04.14 | 归档环境搭建笔记与 Day1 五元组理解 |
| 2026.04.15 | 新增 `Q-table.py`，实现 ε-greedy 策略及完整测试脚本 |
| 2026.04.16 | 新增 `Q_learning_update.py`，完成贝尔曼更新公式代码落地 |
| 2026.04.17 | 完成完整 Q-learning 训练循环，加入调参实验与可视化 |
| 2026.04.18 | 整理 README，归档调参实验核心结论 |

---

## 👤 作者

**陈韬** · 东华大学机械工程 2025 级

长期目标：多智能体强化学习 × 具身智能 × 机器人/汽车科技

> 这是 RL 地基学习计划的第一个完整闭环项目。两周目标：独立手写 DQN，让 CartPole 稳定收敛。
