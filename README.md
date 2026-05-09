
## 开篇:我遇到的问题

我身处在一个AI盛行的时代，身边的人都说，不会用AI将来将会被淘汰。所以我整个的但智能体强化学习都是利用豆包和Claude帮助我一步步学习的。Claude可以帮助我明确方向，同时在我拖延与偏离的时候拉回我。然后是豆包，利用它的专家模式一步步学会这些困难的问题。
但是当我开始写这篇文章的时候，我发现我是最后才弄懂Q-Learning与DQN之间到底有什么区别。也遇到一些问题。在开始新篇章DQN的时候，我还是带着原来的Q-Learning思维，在用到的代码中都会加上一个Q_table.但是每次都会被豆包批评。然后我就不再纠结，虽然有很多一些基础性的知识我都会问豆包 。但是这没关系。与我喜欢的一部动漫的主人公一样简单的挥拳都会挥上100完次。直到我写完DQN（CartPole-v1)并看到小车倒立摆的游戏的时候，我才意识到为什么使用DQN。但是这些都不是问题，问题在于我今天在写这篇文章的时候，才知道两者的区别。

> [!NOTE]
> 我一直以为，状态越多越难。但 FrozenLake 有 16 个状态，Q-Learning 轻松跑通；CartPole 只有 4 个状态，Q-Learning 却完全用不了。这个矛盾困扰了我很久，直到今天我才真正想明白。

---

## 一、卡死我的核心难题：维数灾难

1. 一句话说清楚：什么是维数灾难？
	1. 维数灾难：在智能体的场景中，每增加一个状态维度就会导致状态数量的指数级爆炸。简单来说就是：牵一发以动全身。
		如果我硬要把cartpole的4个连续的状态离散化，每个维度拆成100个区间，总状态数就是100^4=1亿个，Q表大小就是1^2亿=2亿个单元格。我的笔记本电脑内存只有16GB，根本存不了这么大的表格，更别说什么更新。
	2. 状态维度：用来完整描述智能体当前状态的【独立特征数量】
	3. 在FrozenLake-v1中是4✖️4网格，一个格子对应一个状态。我们描述智能体当前的状态只需要一个独立特征（当前在哪一个格子）（0-15号），所以状态维度为1。在另一个项目CartPole-v1中，状态维度为4.（1.小车的水平位置。2.杆摆动的角速度。3.小车的移动速度。4.杆子偏离竖直方向的角度）。
	4. Q-learning更像是一张Excel表格，只能存储有限的数据。同时在每个状态的时候要查看表格，根据表中的方法来决策。然而DQN通过输入当前的状态，再通过一些函数等等计算得出决策后执行动作，可以处理更多的数据与状态。
2. 为什么 FrozenLake 能用 Q-Learning 跑通，CartPole 不行？
	1. 因为CartPole的状态是连续实数，理论上有无限个状态，根本无法枚举。而Q-learning的状态是离散的，有限的，可枚举的。这是“有限”与“无限”的本质区别。
	2. Q-Learning：
  ```
  q_table=np.zeros([5,3])  
  q_table[1]=[0.1,0.9,0.5]
  ```


---

## 二、两条完全不同的路：查表 vs 拟合
### 第一张图：Q-Learning = 一张 Excel 查分表
#### Q-Learning的核心是提前存储每个状态的Q值，遇到状态直接查表。
#### 本质是“死记硬背”

<img width="514" height="572" alt="Screenshot 2026-05-07 125110" src="https://github.com/user-attachments/assets/22e1b907-5266-4b5f-b823-8515a1d66d04" />
<img width="1024" height="1024" alt="Q-Learning（FrozenLake-v1）" src="https://github.com/user-attachments/assets/93a8ebb8-819b-4936-8d75-dff33531aa00" />


### 第二张图：DQN(CartPole-v1)=一个黑盒子函数
#### DQN的核心是学习一个输入状态、输出Q值的函数，不需要提前存储所有状态。

<img width="604" height="460" alt="Screenshot 2026-05-07 124015" src="https://github.com/user-attachments/assets/19ec3f95-da8b-4be1-9c83-e9cfb74c622b" />
<img width="2004" height="1334" alt="Screenshot 2026-05-07 112039" src="https://github.com/user-attachments/assets/fafd5820-cde2-4903-891e-fbadaddc588d" />
 <img width="2731" height="1535" alt="LEARNING" src="https://github.com/user-attachments/assets/13e3afbb-2a0a-456d-a787-23d9662d1837" />



--- 
## 函数逼近：从“死记硬背”到“学会规律”
### DQN 用什么方法，解决了 Q-Learning 解决不了的维数灾难问题？
####  通过函数逼近计算，通过每一个状态来相应的计算每一次的输出。



```
	class SimpleDQN(nn.Module): def __init__(self): 
	super().__init__() 
	self.fc1 = nn.Linear(4, 128) # 输入层：4个状态 
	self.fc2 = nn.Linear(128, 128) # 隐藏层 
	self.fc3 = nn.Linear(128, 2) # 输出层：2个动作的Q值 
	def forward(self, x): 
	x = torch.relu(self.fc1(x)) 
	x = torch.relu(self.fc2(x)) 
	x = self.fc3(x) return x
```
#### 同时不需要存储所有状态，就能输出Q值。
 代码中如何使用函数逼近？
定义网络（逼近器）：
```
online_net = SimpleDQN()  # 创建在线网络，输入4个状态，输出2个Q值
target_net = SimpleDQN()  # 目标网络，稳定训练
```
SimpleDQN是nn.Module子类，里面有全连接层（fc1, fc2, fc3），这就是逼近函数：f(state) ≈ Q_values。
预测Q值（利用逼近）：
```
def epsilon_greedy_action(online_net, state, epsilon, n_actions):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_value = online_net(state_tensor)  # 网络逼近：输入状态，输出[ Q_left, Q_right ]
    action = q_value.argmax(dim=1).item()  # 选Q值最大的动作
    # ... 然后epsilon判断是否探索
```
这里用网络“逼近”Q值：给状态，网络算出Q_left和Q_right。不是查表，是算出来的近似值。

> 问题：传统Q-learning用表格存Q值（每个状态-动作对一个值），但如果状态空间大（比如连续的，如小车位置无限），表格存不下（内存爆炸）。
解决方案：用函数逼近！用一个模型（通常神经网络）来“猜”Q值：输入状态，输出所有动作的Q值。不用存所有，算出来。
比喻：表格像字典，存每个词的解释。但词太多，存不下。你训练一个AI模型（函数），输入词，输出解释——逼近字典的功能。

**我的 SimpleDQN 网络只有 `4×128 + 128×128 + 128×2 = 17024` 个可训练参数。只用这 1.7 万个参数，就能拟合出 CartPole 无限个状态对应的 Q 值，这就是函数逼近的魔力。**

--- 
## 结尾：我的复盘与下一步
  这次深度思考，让我收获了许多，最大的收获就是理解了Q-Learning与DQN的区别，方便我以后应用这两个方法。
  之前是“跑通了代码”，现在终于理解原来代码之间可以有这么多的讲究。比如未来折扣，探索概率等等。这些是我以前没有考虑过的。但是现在给我更多的想法与见解。
  说到训练CartPole时，奖励回退问题。我想原因在于前期的时候，多探索确实可以提升奖励值，但是在 利用时，选择的最大奖励值的动作可能存在偏差，使奖励下跌。就是我们说的奖励回退现象。
  接下来我要学习的连续动作控制算法（PPO），不能用DQN直接做，因为DQN的输出是有限的，无法应对无限个连续动作的情况。
