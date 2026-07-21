# Isaac 环境适配器接口说明

本文档目标是让 Isaac 环境可以被 `isaac_collect_z_dataset_map.py` 调用，用于采集 TD-MPC2 encoder 的 z 隐变量数据集。

核心要求很简单：**每个 step 返回每个机器人的 `laser_map + velocity + goal_rel`**。

```python
{
    "laser_map": np.ndarray,        # [128, 256]，局部激光地图图像，不是 180 维 lidar scan
    "velocity": np.ndarray,         # [v, w]，当前机器人真实速度，单位 m/s 和 rad/s
    "goal_rel": np.ndarray,         # [distance, theta]，机器人坐标系下目标极坐标
}
```

采集脚本会负责：

```text
Isaac obs
  -> map / velocity / goal_rel 解析
  -> 速度与目标归一化
  -> 构造 temporal map / dual-branch map 输入
  -> 调用 TD-MPC2 WorldModel.fe(...)
  -> 得到 z
  -> 保存为 .pt 数据集
```

---

## 1. 推荐文件位置

建议在 Isaac 工程中放置如下结构：

```text
<Isaac工程>/
├── isaac_env_adapter/
│   ├── __init__.py
│   └── random_nav_env.py
│
├── tdmpc_bridge/
│   ├── isaac_collect_z_dataset_map.py
│   ├── exp_config.py
│   ├── tdmpc2.py
│   ├── tdmpc2_common/
│   ├── tools/
│   └── checkpoints/
│       └── tdmpc2_irsim_trained.pt
│
└── datasets/
    └── isaac_z_dataset/
```

Isaac 环境适配器建议实现为：

```text
<Isaac工程>/isaac_env_adapter/random_nav_env.py
```

里面提供一个类：

```python
class IsaacRandomNavEnv:
    def __init__(self, num_envs=1, num_agents=4, headless=True, **kwargs):
        ...

    def reset(self, seed=None):
        ...

    def step(self, action):
        ...
```

采集脚本运行时会通过下面两个参数导入这个类：

```bash
--isaac-env-module isaac_env_adapter.random_nav_env
--isaac-env-class IsaacRandomNavEnv
```

对应关系是：

```text
module: isaac_env_adapter.random_nav_env
file:   <Isaac工程>/isaac_env_adapter/random_nav_env.py
class:  IsaacRandomNavEnv
```

---

## 2. reset() 返回格式

`reset()` 可以返回：

```python
obs
```

也可以返回：

```python
obs, info
```

### 2.1 多机器人推荐格式

最推荐返回 list，每个机器人一个 dict：

```python
obs = [
    {
        "laser_map": laser_map_0,        # np.ndarray, [128, 256]
        "velocity": np.array([v0, w0], dtype=np.float32),
        "goal_rel": np.array([distance0, theta0], dtype=np.float32),
    },
    {
        "laser_map": laser_map_1,
        "velocity": np.array([v1, w1], dtype=np.float32),
        "goal_rel": np.array([distance1, theta1], dtype=np.float32),
    },
]
```

list 长度应该等于 `num_agents`。

### 2.2 单机器人格式

单机器人也可以直接返回一个 dict：

```python
obs = {
    "laser_map": laser_map,                  # [128, 256]
    "velocity": np.array([v, w], dtype=np.float32),
    "goal_rel": np.array([distance, theta], dtype=np.float32),
}
```

### 2.3 batch dict 格式

如果 Isaac 环境更方便返回 batch，也支持：

```python
obs = {
    "laser_map": np.ndarray,     # [num_agents, 128, 256]
    "velocity": np.ndarray,      # [num_agents, 2]
    "goal_rel": np.ndarray,      # [num_agents, 2]
}
```

不过为了减少歧义，推荐用 list-of-dict。

---

## 3. step(action) 输入和返回格式

采集脚本会随机生成动作，然后调用：

```python
ret = env.step(action)
```

### 3.1 action 格式

`action` 的 shape 通常是：

```python
action.shape == [num_agents, 2]
```

每个机器人：

```python
action[i] = [v, w]
```

含义取决于运行采集脚本时的 `--action-format`。

#### 情况 A：`--action-format real`

这是推荐设置。Isaac 环境收到的就是真实速度：

```python
action[i] = [v_mps, w_radps]
```

其中：

```text
v_mps   : 线速度，单位 m/s
w_radps : 角速度，单位 rad/s
```

#### 情况 B：`--action-format norm`

Isaac 环境收到的是归一化动作：

```python
action[i] = [v_norm, w_norm]
```

范围是：

```text
v_norm in [-1, 1]
w_norm in [-1, 1]
```

归一化动作和真实速度的关系：

```python
v_real = ((v_norm + 1.0) / 2.0) * max_linear_vel
w_real = w_norm * max_angular_vel
```

当前原配置里常用默认值：

```text
max_linear_vel  = 4.0
max_angular_vel = 3.0
```

如果没有特殊原因，Isaac 侧建议接收真实速度，也就是运行采集脚本时使用：

```bash
--action-format real
```

### 3.2 step() 返回格式

推荐返回 Gym 老格式：

```python
return obs, reward, done, info
```

也支持 Gymnasium 格式：

```python
return obs, reward, terminated, truncated, info
```

也支持 dict：

```python
return {
    "obs": obs,
    "reward": reward,
    "done": done,
    "info": info,
}
```

其中：

```text
obs    : 下一步观测，格式和 reset() 返回的 obs 一样
reward : z 数据采集阶段不是必须使用，可以先返回 0
done   : bool 或 np.ndarray[bool]，shape [num_agents]
info   : dict，可放调试信息
```

如果暂时没有 reward：

```python
reward = np.zeros(self.num_agents, dtype=np.float32)
```

如果暂时不处理 episode 终止：

```python
done = np.zeros(self.num_agents, dtype=bool)
```

---

## 4. obs["laser_map"] 要求

这是最重要的字段。

Isaac 不要返回 180 维 lidar scan，也不要返回原 IRSim 的 `state_all`。采集脚本需要的是已经生成好的局部激光地图图像。

推荐：

```python
laser_map: np.ndarray
shape: [128, 256]
dtype: np.float32
value range: [0, 1]
```

也可以返回 uint8：

```python
laser_map: np.ndarray
shape: [128, 256]
dtype: np.uint8
value range: [0, 255]
```

采集脚本会自动归一化到 `[0, 1]`。

### 4.1 尺寸约定

TD-MPC2 配置中的图像尺寸是：

```text
img_height = 128
img_width  = 256
```

所以建议 Isaac 直接返回：

```python
laser_map.shape == (128, 256)
```

注意顺序是：

```text
[H, W] = [128, 256]
```

不是：

```text
[W, H] = [256, 128]
```

脚本虽然会尝试自动转置或 resize，但最好 Isaac 侧直接输出正确尺寸。

### 4.2 地图值语义

推荐语义：

```text
1.0 = free / 可通行区域
0.0 = obstacle / 障碍物区域
```

如果 Isaac 侧生成的是相反语义：

```text
1.0 = obstacle
0.0 = free
```

可以在运行采集脚本时加：

```bash
--map-invert
```

但最好提前确认语义，避免生成的 z 数据集整体偏移。

### 4.3 地图坐标方向建议

建议局部地图和训练时保持一致：

```text
机器人位于图像底部中心附近
机器人前方朝图像上方
机器人左侧对应图像左侧
机器人右侧对应图像右侧
```

也就是说，图像表达的是“以当前机器人为中心/自车视角”的局部地图，而不是直接给全局地图。

多机器人时，每个 agent 的 `laser_map` 都应该是该 agent 自己的局部视角。

---

## 5. 是否需要返回历史地图

### 5.1 最简单方式：每步只返回单帧

最推荐先实现这个：

```python
"laser_map": np.ndarray  # [128, 256]
```

采集脚本会自动维护每个 agent 的历史队列，构造 temporal map。

如果模型需要 `T=8` 帧历史，那么第一步没有历史时，脚本会用当前帧重复填满历史：

```text
[t, t, t, t, t, t, t, t]
```

后续会变成最近 T 帧：

```text
t-7, t-6, ..., t
```

### 5.2 Isaac 直接返回 temporal map

如果 Isaac 侧已经能返回历史图，也可以：

```python
"laser_map": np.ndarray  # [T, 128, 256]
```

通常：

```text
T = cfg.sample_length = 8
```

如果 T 太长，脚本取最后 T 帧；如果 T 太短，脚本会在前面补第一帧。

### 5.3 dual-branch 格式

如果模型是 dual-branch encoder，也可以显式返回：

```python
"laser_map": {
    "combine": np.ndarray,      # [128, 256]
    "temporal": np.ndarray,     # [T, 128, 256]
}
```

或者返回：

```python
{
    "combine_map": np.ndarray,      # [128, 256]
    "temporal_map": np.ndarray,     # [T, 128, 256]
    "velocity": np.ndarray,         # [2]
    "goal_rel": np.ndarray,         # [2]
}
```

不过不是必须。只返回单帧 `[128, 256]` 时，采集脚本也能自动构造 temporal 输入。

---

## 6. obs["velocity"] 要求

推荐 Isaac 返回当前机器人真实速度：

```python
"velocity": np.array([v, w], dtype=np.float32)
```

含义：

```text
v : 当前机器人线速度，单位 m/s
w : 当前机器人角速度，单位 rad/s
```

运行采集脚本时默认使用：

```bash
--velocity-format real
```

脚本会把真实速度转成模型输入格式：

```python
v_model = v / max_linear_vel
w_model = (w / max_angular_vel + 1.0) / 2.0
```

例如：

```python
velocity = [2.0, 0.0]
max_linear_vel = 4.0
max_angular_vel = 3.0
```

则模型输入为：

```python
[v_model, w_model] = [0.5, 0.5]
```

注意：最好返回机器人当前实际速度，而不是刚刚发送的 action command。如果暂时拿不到真实速度，可以先返回上一条 cmd velocity，但请在 `info` 或代码注释里说明。

如果 Isaac 已经返回模型归一化速度，则运行脚本时需要改成：

```bash
--velocity-format model_norm
```

如果 Isaac 返回的是动作归一化速度 `[-1, 1]`，则运行脚本时改成：

```bash
--velocity-format action_norm
```

---

## 7. obs["goal_rel"] 要求

推荐 Isaac 返回机器人坐标系下的目标极坐标：

```python
"goal_rel": np.array([distance, theta], dtype=np.float32)
```

含义：

```text
distance : 当前机器人到目标点的距离，单位 m
theta    : 当前机器人朝向下，目标相对机器人正前方的夹角，单位 rad
```

角度范围建议：

```text
theta in [-pi, pi]
```

坐标约定：

```text
theta = 0   : 目标在机器人正前方
theta > 0   : 目标在机器人左侧
theta < 0   : 目标在机器人右侧
```

运行采集脚本时默认可用：

```bash
--goal-rel-key goal_rel
--goal-rel-format polar
```

### 7.1 如果更方便返回局部 xy

也可以返回机器人坐标系下的局部目标 xy：

```python
"goal_xy": np.array([gx, gy], dtype=np.float32)
```

约定：

```text
gx : 机器人前方为正
gy : 机器人左侧为正
```

运行脚本时可以使用：

```bash
--goal-rel-format xy
```

### 7.2 如果更方便返回全局 pose + goal

也可以不返回 `goal_rel`，改为返回：

```python
"pose": np.array([x, y, yaw], dtype=np.float32)
"goal": np.array([goal_x, goal_y], dtype=np.float32)
```

采集脚本会自动计算机器人局部坐标系下的目标关系。

全局 pose 的 yaw 单位是 rad。

---

## 8. 可选字段

下面这些字段不是 encoder 的最小输入，但建议返回，方便之后检查数据质量：

```python
{
    "pose": np.array([x, y, yaw], dtype=np.float32),
    "goal": np.array([goal_x, goal_y], dtype=np.float32),
    "collision": bool,
    "arrive": bool,
    "robot_id": int,
    "min_dist": float,
}
```

采集脚本会把部分字段保存到样本的 `extra` 里。

如果训练这个 checkpoint 时打开了 `deviation_mode=True`，还需要返回路径偏差信息：

```python
"path": np.array([path_deviation_norm, path_angle_norm], dtype=np.float32)
```

如果训练这个 checkpoint 时打开了 `no_goal=True` 且 `progress_ratio_mode=True`，还需要返回：

```python
"progress": np.array([...], dtype=np.float32)
```

如果暂时没有这些字段，采集脚本默认可以用 0 填充；但如果原 checkpoint 确实依赖这些输入，最好 Isaac 侧提供真实值。

---

## 9. 最小可工作的 Isaac adapter 骨架

可以从下面这个文件骨架开始实现：

```python
# <Isaac工程>/isaac_env_adapter/random_nav_env.py

import numpy as np


class IsaacRandomNavEnv:
    def __init__(self, num_envs=1, num_agents=4, headless=True, **kwargs):
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.headless = bool(headless)

        # TODO: 初始化 Isaac Sim / Isaac Lab 环境
        # self.sim = ...
        # self.robots = ...
        # self.goals = ...

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # TODO:
        # 1. reset Isaac world
        # 2. 随机初始化机器人位置、朝向
        # 3. 随机初始化目标点
        # 4. 推进一小步，确保传感器和状态数据可读

        return self._get_obs_all()

    def step(self, action):
        """
        action shape: [num_agents, 2]

        如果采集脚本使用 --action-format real:
            action[i] = [v_mps, w_radps]

        如果采集脚本使用 --action-format norm:
            action[i] = [v_norm, w_norm]
        """
        action = np.asarray(action, dtype=np.float32)

        # TODO: 把 action 发给每个机器人
        # for i in range(self.num_agents):
        #     v, w = action[i]
        #     self.set_robot_velocity(i, v, w)

        # TODO: 推进 Isaac 仿真一步
        # self.sim.step()

        obs = self._get_obs_all()
        reward = np.zeros(self.num_agents, dtype=np.float32)
        done = np.zeros(self.num_agents, dtype=bool)
        info = {}

        return obs, reward, done, info

    def _get_obs_all(self):
        return [self._get_obs_one(i) for i in range(self.num_agents)]

    def _get_obs_one(self, i):
        laser_map = self._build_laser_map(i)           # [128, 256], float32, 0~1
        velocity = self._get_robot_velocity(i)         # [v, w]
        goal_rel = self._get_goal_relative_polar(i)    # [distance, theta]

        return {
            "laser_map": laser_map.astype(np.float32),
            "velocity": np.asarray(velocity, dtype=np.float32),
            "goal_rel": np.asarray(goal_rel, dtype=np.float32),
        }

    def _build_laser_map(self, i):
        # TODO: 替换成 Isaac 侧真实局部激光地图生成逻辑
        # 必须返回 [128, 256]
        # 推荐语义: 1.0=free, 0.0=obstacle
        return np.ones((128, 256), dtype=np.float32)

    def _get_robot_velocity(self, i):
        # TODO: 返回第 i 个机器人当前真实速度 [m/s, rad/s]
        return np.array([0.0, 0.0], dtype=np.float32)

    def _get_goal_relative_polar(self, i):
        # TODO: 返回机器人坐标系下目标关系 [distance, theta]
        return np.array([3.0, 0.0], dtype=np.float32)
```

---

## 10. 运行采集脚本示例

假设 Isaac 工程路径是：

```bash
export ISAAC_PROJECT=/media/nav/18a974f3-ee2a-474a-8c7b-8ee5b345bb1d10/wheel-arm/isaac_ws/isaac_work-master/scripts
export TDMPC_BRIDGE=$ISAAC_PROJECT/tdmpc_bridge
export PYTHONPATH=$ISAAC_PROJECT:$TDMPC_BRIDGE:$PYTHONPATH
```

普通 Python 运行示例：

```bash
python $TDMPC_BRIDGE/isaac_collect_z_dataset_map.py \
  --project-root $TDMPC_BRIDGE \
  --model-path $TDMPC_BRIDGE/checkpoints/801 \
  --output $ISAAC_PROJECT/datasets/isaac_z_dataset_v1 \
  --isaac-env-module isaac_env_adapter.random_nav_env \
  --isaac-env-class IsaacRandomNavEnv \
  --isaac-env-kwargs-json '{"num_envs":1,"num_agents":4,"headless":true}' \
  --num-agents 4 \
  --num-episodes 200 \
  --episode-steps 150 \
  --gpu 0 \
  --map-key laser_map \
  --map-input-mode combine \
  --map-normalize auto \
  --velocity-key velocity \
  --velocity-format real \
  --goal-rel-key goal_rel \
  --goal-rel-format polar \
  --action-format real \
  --chunk-size 50000 \
  --print-every 1000
```

如果使用 Isaac Sim 自带 Python，替换第一行执行器即可，例如：

```bash
/path/to/isaac-sim/python.sh $TDMPC_BRIDGE/isaac_collect_z_dataset_map.py \
  --project-root $TDMPC_BRIDGE \
  --model-path $TDMPC_BRIDGE/checkpoints/tdmpc2_irsim_trained.pt \
  --output $ISAAC_PROJECT/datasets/isaac_z_dataset_v1 \
  --isaac-env-module isaac_env_adapter.random_nav_env \
  --isaac-env-class IsaacRandomNavEnv \
  --isaac-env-kwargs-json '{"num_envs":1,"num_agents":4,"headless":true}' \
  --num-agents 4 \
  --num-episodes 200 \
  --episode-steps 150 \
  --gpu 0 \
  --map-key laser_map \
  --map-input-mode combine \
  --velocity-format real \
  --goal-rel-format polar \
  --action-format real
```

如果使用 Isaac Lab，通常类似：

```bash
/path/to/IsaacLab/isaaclab.sh -p $TDMPC_BRIDGE/isaac_collect_z_dataset_map.py \
  --project-root $TDMPC_BRIDGE \
  --model-path $TDMPC_BRIDGE/checkpoints/tdmpc2_irsim_trained.pt \
  --output $ISAAC_PROJECT/datasets/isaac_z_dataset_v1 \
  --isaac-env-module isaac_env_adapter.random_nav_env \
  --isaac-env-class IsaacRandomNavEnv \
  --isaac-env-kwargs-json '{"num_envs":1,"num_agents":4,"headless":true}' \
  --num-agents 4 \
  --num-episodes 200 \
  --episode-steps 150 \
  --gpu 0 \
  --map-key laser_map \
  --map-input-mode combine \
  --velocity-format real \
  --goal-rel-format polar \
  --action-format real
```

---

## 11. 采集脚本参数和 obs 字段对应关系

| 采集脚本参数 | Isaac obs 字段 | 说明 |
|---|---|---|
| `--map-key laser_map` | `obs["laser_map"]` | 局部激光地图，推荐 `[128,256]` |
| `--velocity-key velocity` | `obs["velocity"]` | 当前速度 `[v,w]` |
| `--goal-rel-key goal_rel` | `obs["goal_rel"]` | 目标极坐标 `[distance,theta]` |
| `--pose-key pose` | `obs["pose"]` | 可选，全局机器人位姿 `[x,y,yaw]` |
| `--goal-key goal` | `obs["goal"]` | 可选，全局目标 `[gx,gy]` |
| `--path-key path` | `obs["path"]` | 可选，路径偏差输入 |
| `--progress-key progress` | `obs["progress"]` | 可选，进度输入 |

---

## 12. 快速自检代码

实现好 `IsaacRandomNavEnv` 后，可以先单独检查 obs：

```python
import numpy as np
from isaac_env_adapter.random_nav_env import IsaacRandomNavEnv


env = IsaacRandomNavEnv(num_envs=1, num_agents=4, headless=True)
obs = env.reset()

assert isinstance(obs, list), type(obs)
assert len(obs) == 4

for i, o in enumerate(obs):
    assert "laser_map" in o, f"agent {i} missing laser_map"
    assert "velocity" in o, f"agent {i} missing velocity"
    assert "goal_rel" in o, f"agent {i} missing goal_rel"

    m = np.asarray(o["laser_map"])
    v = np.asarray(o["velocity"])
    g = np.asarray(o["goal_rel"])

    assert m.shape == (128, 256), f"agent {i} map shape wrong: {m.shape}"
    assert v.shape == (2,), f"agent {i} velocity shape wrong: {v.shape}"
    assert g.shape == (2,), f"agent {i} goal_rel shape wrong: {g.shape}"
    assert np.isfinite(m).all(), f"agent {i} map has nan/inf"
    assert np.isfinite(v).all(), f"agent {i} velocity has nan/inf"
    assert np.isfinite(g).all(), f"agent {i} goal_rel has nan/inf"

print("Isaac obs interface check passed")
```

也可以检查 step：

```python
action = np.zeros((4, 2), dtype=np.float32)
next_obs, reward, done, info = env.step(action)
print(type(next_obs), np.asarray(reward).shape, np.asarray(done).shape, type(info))
```

---

## 13. 常见错误

### 13.1 把 180 维 lidar scan 当成 laser_map 返回

错误：

```python
"laser_map": np.ndarray  # [180]
```

正确：

```python
"laser_map": np.ndarray  # [128, 256]
```

本项目的 Isaac 采集流程不再调用 IRSim 的 `State_Interpreter`，所以不要返回 180 维 scan。

### 13.2 地图黑白语义反了

如果发现生成的数据不对，优先检查：

```text
训练时: 1=free, 0=obstacle
Isaac:  是否也是 1=free, 0=obstacle
```

如果 Isaac 是反的，运行脚本加：

```bash
--map-invert
```

### 13.3 地图不是局部自车坐标

每个 agent 的 `laser_map` 应该是该 agent 自己的局部视角，不是直接全局地图。

多机器人时不能把同一张全局地图不经裁剪/旋转就返回给所有机器人。

### 13.4 速度单位不一致

默认 `--velocity-format real` 时，`velocity` 必须是：

```text
m/s, rad/s
```

不要返回已经归一化过的速度，除非运行脚本时使用：

```bash
--velocity-format model_norm
```

### 13.5 目标角度方向反了

推荐约定：

```text
theta > 0 表示目标在机器人左侧
theta < 0 表示目标在机器人右侧
```

如果 Isaac 使用相反方向，需要在 Isaac 侧修正，或者在 adapter 里转换。

### 13.6 reset 后第一帧传感器未更新

如果 reset 后传感器数据为空或全 0，建议 reset 后推进一小步仿真，再读取 obs。

---

## 14. 最短版要求

只要先保证下面这个接口正确，采集脚本就可以跑起来：

```python
def reset(self, seed=None):
    return [
        {
            "laser_map": np.ones((128, 256), dtype=np.float32),
            "velocity": np.array([0.0, 0.0], dtype=np.float32),
            "goal_rel": np.array([3.0, 0.0], dtype=np.float32),
        }
        for _ in range(self.num_agents)
    ]


def step(self, action):
    obs = self._get_obs_all()
    reward = np.zeros(self.num_agents, dtype=np.float32)
    done = np.zeros(self.num_agents, dtype=bool)
    info = {}
    return obs, reward, done, info
```

然后逐步把里面的假数据替换成 Isaac 的真实局部地图、真实速度和真实目标关系。
