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
│   └── daily_note/
│       ├── day_env_setup_note.md # 环境搭建笔记
│       └── day1_understanding_of_Five-Tuple.md  # Day1 RL五元组理解

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

## 🗓️ 更新记录

| 日期 | 内容 |
|------|------|
| 2026.4.13 | 新增 `marl_demo.py`，跑通 PettingZoo MPE 追逐环境 |
| 2026.4.14 | 新增 `FrozenLake_v1.py`，完成 Gymnasium 基础交互循环 |
| 2026.4.14 | 归档环境搭建笔记与 Day1 五元组理解 |
