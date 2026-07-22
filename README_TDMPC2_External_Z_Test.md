# Isaac 测试脚本说明：`isaac_test_tdmpc2_external_z.py`

本文档说明如何在 Isaac 场景中测试 TD-MPC2，其中 **TD-MPC2 的 dynamics / reward / value / policy / planner 仍然使用原 IRSim 工程训练好的 checkpoint**，但是 **观测到隐变量 `z` 的编码不再使用原 TD-MPC2 encoder**，而是使用已经训练好的：

```text
Isaac 观测 obs
    -> 大 encoder，frozen
    -> 小 adapter
    -> 隐变量 z
```

然后测试脚本使用这个外部 `z` 进入 TD-MPC2 的 latent planner：

```text
Isaac obs_t
    -> external_z_encoder(obs_t)
    -> z_t
    -> TD-MPC2 planner / dynamics / reward / Q / policy
    -> action_t
    -> Isaac env.step(action_t)
```

也就是说，这个测试脚本的核心目标是：

> **验证“Isaac 大 encoder + 小 adapter”输出的 z，能不能接上原 TD-MPC2 checkpoint 的后半部分，在 Isaac 场景里完成导航测试。**

---

## 1. 相关文件建议放置位置

建议在 Isaac 工程中使用如下结构：

```text
<Isaac工程>/
├── tdmpc_bridge/
│   ├── isaac_test_tdmpc2_external_z.py
│   ├── exp_config.py
│   ├── tdmpc2.py
│   ├── tdmpc2_common/
│   │   ├── __init__.py
│   │   ├── world_model.py
│   │   ├── layers.py
│   │   ├── math.py
│   │   ├── init.py
│   │   ├── scale.py
│   │   └── ...
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── utils_network.py
│   │   └── ...
│   └── checkpoints/
│       └── tdmpc2_irsim_trained.pt
│
├── isaac_env_adapter/
│   ├── __init__.py
│   └── random_nav_env.py
│
├── isaac_z_adapter/
│   ├── __init__.py
│   └── external_z_encoder.py
│
└── test_runs/
    └── external_z_tdmpc2_test_001/
```

其中：

```text
tdmpc_bridge/
```

放原 TD-MPC2 工程中复制过来的模型推理代码，以及测试脚本。

```text
isaac_env_adapter/
```

放 Isaac 环境适配器，也就是 `reset()` / `step(action)` 的环境类。

```text
isaac_z_adapter/
```

放“大 encoder + 小 adapter”的接口文件。

---

## 2. 本脚本和原 TD-MPC2 流程的区别

原来的 TD-MPC2 推理大致是：

```text
obs
    -> TD-MPC2 自己的 encoder / multi_encode
    -> z
    -> plan(z)
    -> action
```

现在测试时改成：

```text
obs
    -> 外部 encoder + adapter
    -> z
    -> plan(z)
    -> action
```

因此：

```text
会加载原 TD-MPC2 checkpoint：是的
会使用原 dynamics / reward / Q / policy / MPPI planner：是的
会使用原 TD-MPC2 encoder / multi_encode：不会
会训练网络：不会，只做测试推理
```

脚本内部不会调用原来的：

```python
TDMPC2.act(obs, ...)
```

因为 `act()` 会重新走原始 encoder。脚本中使用了一个新的 controller，从外部 `z` 开始调用 TD-MPC2 的 `plan(z, ...)`。

---

## 3. 需要提供的内容

需要提供以下文件或目录给 Isaac 测试工程：

```text
<Isaac工程>/tdmpc_bridge/isaac_test_tdmpc2_external_z.py
<Isaac工程>/tdmpc_bridge/exp_config.py
<Isaac工程>/tdmpc_bridge/tdmpc2.py
<Isaac工程>/tdmpc_bridge/tdmpc2_common/
<Isaac工程>/tdmpc_bridge/tools/
<Isaac工程>/tdmpc_bridge/checkpoints/tdmpc2_irsim_trained.pt
```

注意：

1. `exp_config.py` 必须尽量使用训练这个 TD-MPC2 checkpoint 时的同一份配置。
2. checkpoint 通常只保存 `model.state_dict()`，不保存完整 config。
3. 如果 config 和 checkpoint 不匹配，可能会出现 `load_state_dict` shape mismatch，或者能加载但测试效果不可信。
4. 原 TD-MPC2 代码中有较多 `.cuda()` 和 `torch.device("cuda")`，因此测试建议在有 CUDA 的 Isaac 环境中运行。

---

## 4. 负责 Isaac 的同学需要做什么

负责 Isaac 的同学主要需要提供两类接口：

```text
1. Isaac 环境接口
2. 外部 z encoder 接口
```

---

# Part A：Isaac 环境接口

## 5. Isaac env 文件位置和类名

建议文件位置：

```text
<Isaac工程>/isaac_env_adapter/random_nav_env.py
```

其中提供一个类，例如：

```python
class IsaacRandomNavEnv:
    def __init__(self, num_envs=1, num_agents=4, headless=True, **kwargs):
        ...

    def reset(self, seed=None):
        ...

    def step(self, action):
        ...
```

运行测试脚本时用下面两个参数导入：

```bash
--isaac-env-module isaac_env_adapter.random_nav_env
--isaac-env-class IsaacRandomNavEnv
```

对应关系是：

```text
isaac_env_adapter.random_nav_env
    -> <Isaac工程>/isaac_env_adapter/random_nav_env.py

IsaacRandomNavEnv
    -> random_nav_env.py 文件中的类名
```

---

## 6. `reset()` 返回格式

`reset()` 可以返回：

```python
obs
```

也可以返回：

```python
obs, info
```

推荐多机器人返回 list，每个机器人一个 observation dict：

```python
obs = [
    {
        "laser_map": laser_map_0,
        "velocity": np.array([v0, w0], dtype=np.float32),
        "goal_rel": np.array([distance0, theta0], dtype=np.float32),
    },
    {
        "laser_map": laser_map_1,
        "velocity": np.array([v1, w1], dtype=np.float32),
        "goal_rel": np.array([distance1, theta1], dtype=np.float32),
    },
    ...
]
```

其中 list 长度等于 `num_agents`。

单机器人时也可以直接返回一个 dict：

```python
obs = {
    "laser_map": laser_map,
    "velocity": np.array([v, w], dtype=np.float32),
    "goal_rel": np.array([distance, theta], dtype=np.float32),
}
```

也支持 batch dict：

```python
obs = {
    "laser_map": np.ndarray,   # shape [num_agents, 128, 256]
    "velocity": np.ndarray,   # shape [num_agents, 2]
    "goal_rel": np.ndarray,   # shape [num_agents, 2]
}
```

---

## 7. `step(action)` 输入格式

测试脚本每一步会调用：

```python
next_obs, reward, done, info = env.step(action)
```

传入的 `action` 格式由命令行参数决定。

### 7.1 推荐方式：真实速度

如果运行脚本时使用：

```bash
--action-format real
```

那么 Isaac env 收到的是真实速度：

```python
action.shape == [num_agents, 2]
action[i] = [v_mps, w_radps]
```

其中：

```text
v_mps   : 线速度，单位 m/s
w_radps : 角速度，单位 rad/s
```

### 7.2 可选方式：归一化动作

如果运行脚本时使用：

```bash
--action-format norm
```

那么 Isaac env 收到的是 TD-MPC2 原始归一化动作：

```python
action[i] = [v_norm, w_norm]
```

范围是：

```text
v_norm in [-1, 1]
w_norm in [-1, 1]
```

归一化动作和真实速度的关系是：

```python
v_real = ((v_norm + 1.0) / 2.0) * max_linear_vel
w_real = w_norm * max_angular_vel
```

默认配置中通常是：

```text
max_linear_vel  = 4.0
max_angular_vel = 3.0
```

---

## 8. `step(action)` 返回格式

推荐返回 Gym 老格式：

```python
return obs, reward, done, info
```

也支持 Gymnasium 新格式：

```python
return obs, reward, terminated, truncated, info
```

字段含义：

```text
obs    : 下一步观测，格式和 reset() 一样
reward : float 或 np.ndarray[num_agents]
done   : bool 或 np.ndarray[num_agents]
info   : dict
```

如果暂时没有 reward，可以返回 0：

```python
reward = np.zeros(self.num_agents, dtype=np.float32)
```

如果暂时不处理终止，可以返回全 False：

```python
done = np.zeros(self.num_agents, dtype=bool)
```

建议 `info` 或每个 agent 的 obs 中包含成功和碰撞标志，方便测试脚本统计指标：

```python
info = {
    "success": np.array([...], dtype=bool),
    "collision": np.array([...], dtype=bool),
}
```

或者每个 agent obs 中包含：

```python
{
    "success": bool,
    "collision": bool,
}
```

脚本会识别这些字段名：

```text
success / arrive / arrival / reached_goal / is_success
collision / collide / crash / is_collision
```

---

## 9. Isaac obs 必须包含哪些字段

虽然测试脚本本身不再关心外部 encoder 怎么处理 obs，但为了统一接口，推荐每个 agent 的 obs 至少包含：

```python
{
    "laser_map": np.ndarray,        # [128, 256] 或 [T, 128, 256]
    "velocity": np.ndarray,         # [v, w]
    "goal_rel": np.ndarray,         # [distance, theta]
}
```

### 9.1 `laser_map`

`laser_map` 是 Isaac 生成的局部激光地图图像，不是 180 维 lidar scan。

推荐格式：

```python
laser_map.shape == [128, 256]
laser_map.dtype == np.float32
laser_map range == [0, 1]
```

也可以返回 `uint8`：

```python
laser_map.shape == [128, 256]
laser_map.dtype == np.uint8
laser_map range == [0, 255]
```

建议语义：

```text
1.0 = free，可通行区域
0.0 = obstacle，障碍物区域
```

如果你们的地图语义相反，可以运行脚本时加：

```bash
--map-invert
```

注意：在本测试脚本中，`laser_map` 有两个用途：

```text
1. 传给外部 z encoder，由负责 Isaac 的同学自己使用
2. 如果 cfg.use_env_dyn_weight=True，则脚本会用 laser_map 历史帧构造动态区域图，供 TD-MPC2 risk-aware planner 使用
```

如果不想使用第二个用途，可以运行时加：

```bash
--disable-env-dyn-weight
```

### 9.2 `velocity`

推荐返回真实速度：

```python
"velocity": np.array([v, w], dtype=np.float32)
```

其中：

```text
v: 当前机器人真实线速度，单位 m/s
w: 当前机器人真实角速度，单位 rad/s
```

运行脚本时默认使用：

```bash
--velocity-format real
```

测试脚本会内部转换成原 TD-MPC2 planner 需要的归一化速度：

```python
v_model = v / max_linear_vel
w_model = (w / max_angular_vel + 1.0) / 2.0
```

如果 Isaac env 已经返回的是上面这种模型归一化速度，则运行时用：

```bash
--velocity-format model_norm
```

如果 Isaac env 返回的是动作归一化格式 `[-1, 1]`，则运行时用：

```bash
--velocity-format action_norm
```

注意：`velocity` 很重要。它不仅可能被外部 encoder 使用，也会被 TD-MPC2 planner 用于速度/加速度约束。最好返回机器人当前实际速度，而不是随便填 0。

### 9.3 `goal_rel`

推荐返回机器人坐标系下的目标极坐标：

```python
"goal_rel": np.array([distance, theta], dtype=np.float32)
```

含义：

```text
distance : 机器人到目标点距离，单位 m
theta    : 目标相对机器人朝向的角度，单位 rad，建议范围 [-pi, pi]
```

坐标约定建议：

```text
theta = 0  表示目标在机器人正前方
theta > 0  表示目标在机器人左侧
theta < 0  表示目标在机器人右侧
```

如果负责 Isaac 的同学的 encoder 需要全局位姿，也可以额外返回：

```python
"pose": np.array([x, y, yaw], dtype=np.float32)
"goal": np.array([goal_x, goal_y], dtype=np.float32)
```

---

# Part B：外部 z encoder 接口

## 10. 外部 z encoder 的职责

负责 Isaac 的同学需要提供一个 Python 接口，它完成：

```text
单个 agent 的 Isaac obs
    -> 大 encoder
    -> 小 adapter
    -> z
```

测试脚本不关心内部怎么做，只要求最后返回的 `z` 可以接到原 TD-MPC2 后半部分。

关键要求：

```text
输入 : 单个 agent 的 obs dict
输出 : z，shape [latent_dim] 或 [1, latent_dim]
类型 : torch.Tensor / np.ndarray / list 都可以
设备 : 可以返回 CPU 或 CUDA tensor，脚本会转到 CUDA float32
```

`z` 的维度必须和原 TD-MPC2 checkpoint 对应：

```python
expected_z_dim = cfg.latent_dim * cfg.enc_num
```

常见情况：

```text
cfg.latent_dim = 512, cfg.enc_num = 1 -> z dim = 512
cfg.latent_dim = 512, cfg.enc_num = 2 -> z dim = 1024
```

如果维度不一致，脚本会打印 warning，但继续运行。维度真正不匹配时，后续 dynamics / reward / Q / policy 很可能报错。

---

## 11. 推荐接口方式一：函数式接口

建议文件位置：

```text
<Isaac工程>/isaac_z_adapter/external_z_encoder.py
```

推荐写法：

```python
import torch
import numpy as np

# 可以在模块全局初始化模型，也可以懒加载
_ENCODER = None


def encode_z(obs, agent_id=None, device=None, cfg=None, **kwargs):
    """
    Args:
        obs:
            单个 agent 的 Isaac observation dict，例如：
            {
                "laser_map": np.ndarray [128,256],
                "velocity": np.ndarray [2],
                "goal_rel": np.ndarray [2],
                ...
            }

        agent_id:
            当前机器人 id，int。

        device:
            测试脚本传入的 torch device，通常是 cuda。

        cfg:
            原 TD-MPC2 的 exp_config 配置对象。

    Returns:
        z:
            torch.Tensor 或 np.ndarray。
            shape [Z] 或 [1, Z]。
    """
    global _ENCODER

    if _ENCODER is None:
        # TODO: 初始化你们的大 encoder + 小 adapter
        # _ENCODER = ...
        # _ENCODER.to(device)
        # _ENCODER.eval()
        pass

    # TODO: 从 obs 中取你们需要的信息
    laser_map = obs["laser_map"]
    velocity = obs.get("velocity", None)
    goal_rel = obs.get("goal_rel", None)

    # TODO: 调用你们的模型得到 z
    # with torch.no_grad():
    #     z = _ENCODER(laser_map, velocity, goal_rel)

    z = torch.zeros(cfg.latent_dim * getattr(cfg, "enc_num", 1), device=device)
    return z
```

运行时使用：

```bash
--z-encoder-module isaac_z_adapter.external_z_encoder
--z-encoder-function encode_z
```

如果函数名就是 `encode_z`，可以不写 `--z-encoder-function`，因为默认就是这个名字。

---

## 12. 推荐接口方式二：类接口

如果你们的大 encoder + adapter 需要 checkpoint，建议用类接口。

文件：

```text
<Isaac工程>/isaac_z_adapter/external_z_encoder.py
```

示例：

```python
import torch
import numpy as np


class IsaacZEncoder:
    def __init__(self, checkpoint_path=None, device="cuda", cfg=None, **kwargs):
        self.device = torch.device(device)
        self.cfg = cfg

        # TODO: 创建大 encoder + adapter
        # self.big_encoder = ...
        # self.adapter = ...

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            # TODO: 按你们自己的保存格式加载
            # self.load_state_dict(ckpt)

        # self.big_encoder.eval()
        # self.adapter.eval()

    @torch.no_grad()
    def __call__(self, obs, agent_id=None, device=None, cfg=None):
        # TODO: 从 obs 中取数据
        laser_map = obs["laser_map"]
        velocity = obs.get("velocity", None)
        goal_rel = obs.get("goal_rel", None)

        # TODO: 调用你们的大 encoder + adapter
        # z = self.adapter(self.big_encoder(...))

        z_dim = self.cfg.latent_dim * getattr(self.cfg, "enc_num", 1)
        z = torch.zeros(1, z_dim, device=self.device)
        return z
```

运行时使用：

```bash
--z-encoder-module isaac_z_adapter.external_z_encoder
--z-encoder-class IsaacZEncoder
--z-encoder-checkpoint /path/to/isaac_big_encoder_adapter.pt
```

测试脚本会尝试把这些参数传给类构造函数：

```python
checkpoint_path
ckpt_path
model_path
device
cfg
```

如果构造函数不接受某些参数，脚本会自动过滤。

---

## 13. 外部 z encoder 返回值支持格式

可以直接返回 Tensor：

```python
return z
```

其中：

```python
z.shape == [Z]
```

或者：

```python
z.shape == [1, Z]
```

也可以返回 dict：

```python
return {"z": z}
```

也可以返回 tuple，脚本会取第一个元素：

```python
return z, debug_info
```

推荐最简单：

```python
return z
```

---

# Part C：运行测试

## 14. 设置环境变量

假设：

```bash
export ISAAC_PROJECT=/home/xxx/isaac_project
export TDMPC_BRIDGE=$ISAAC_PROJECT/tdmpc_bridge
```

运行前设置：

```bash
export PYTHONPATH=$ISAAC_PROJECT:$TDMPC_BRIDGE:$PYTHONPATH
```

这样 Python 才能找到：

```text
isaac_env_adapter.random_nav_env
isaac_z_adapter.external_z_encoder
exp_config.py
tdmpc2.py
tdmpc2_common/
tools/
```

---

## PYL: 构造 z encoder 接口
## 15. 函数式 z encoder 的运行命令示例

```bash
export ISAAC_PROJECT=/media/nav/18a974f3-ee2a-474a-8c7b-8ee5b345bb1d10/wheel-arm/isaac_ws/isaac_work-master/scripts
export TDMPC_BRIDGE=$ISAAC_PROJECT/tdmpc_bridge
export PYTHONPATH=$ISAAC_PROJECT:$TDMPC_BRIDGE:$PYTHONPATH

python $TDMPC_BRIDGE/isaac_test_tdmpc2_external_z.py \
  --project-root $TDMPC_BRIDGE \
  --model-path $TDMPC_BRIDGE/checkpoints/801 \
  --isaac-project-root $ISAAC_PROJECT \
  --isaac-env-module isaac_env_adapter.random_nav_env \
  --isaac-env-class IsaacRandomNavEnv \
  --isaac-env-kwargs-json '{"num_envs":1,"num_agents":4,"headless":true}' \
  --z-encoder-module isaac_z_adapter.external_z_encoder \
  --z-encoder-function encode_z \
  --num-agents 4 \
  --num-episodes 20 \
  --episode-steps 150 \
  --gpu 0 \
  --action-format real \
  --velocity-format real \
  --map-key laser_map \
  --output $ISAAC_PROJECT/test_runs/external_z_tdmpc2_test_001 \
  --save-rollout
```

---

## 16. 类式 z encoder 的运行命令示例

```bash
export ISAAC_PROJECT=/media/nav/18a974f3-ee2a-474a-8c7b-8ee5b345bb1d10/wheel-arm/isaac_ws/isaac_work-master/scripts
export TDMPC_BRIDGE=$ISAAC_PROJECT/tdmpc_bridge
export PYTHONPATH=$ISAAC_PROJECT:$TDMPC_BRIDGE:$PYTHONPATH

python $TDMPC_BRIDGE/isaac_test_tdmpc2_external_z.py \
  --project-root $TDMPC_BRIDGE \
  --model-path $TDMPC_BRIDGE/checkpoints/801 \
  --isaac-project-root $ISAAC_PROJECT \
  --isaac-env-module isaac_env_adapter.random_nav_env \
  --isaac-env-class IsaacRandomNavEnv \
  --isaac-env-kwargs-json '{"num_envs":1,"num_agents":4,"headless":true}' \
  --z-encoder-module isaac_z_adapter.external_z_encoder \
  --z-encoder-class IsaacZEncoder \
  --z-encoder-checkpoint $ISAAC_PROJECT/checkpoints/isaac_big_encoder_adapter.pt \
  --num-agents 4 \
  --num-episodes 20 \
  --episode-steps 150 \
  --gpu 0 \
  --action-format real \
  --velocity-format real \
  --map-key laser_map \
  --output $ISAAC_PROJECT/test_runs/external_z_tdmpc2_test_001 \
  --save-rollout
```

---

## 17. Isaac Sim / Isaac Lab 运行方式

如果 Isaac 环境必须用 Isaac 自带 Python 启动，只需要把上面命令中的 `python` 换掉。

Isaac Sim standalone 示例：

```bash
/path/to/isaac-sim/python.sh $TDMPC_BRIDGE/isaac_test_tdmpc2_external_z.py \
  --project-root $TDMPC_BRIDGE \
  --model-path $TDMPC_BRIDGE/checkpoints/tdmpc2_irsim_trained.pt \
  --isaac-project-root $ISAAC_PROJECT \
  --isaac-env-module isaac_env_adapter.random_nav_env \
  --isaac-env-class IsaacRandomNavEnv \
  --isaac-env-kwargs-json '{"num_envs":1,"num_agents":4,"headless":true}' \
  --z-encoder-module isaac_z_adapter.external_z_encoder \
  --z-encoder-class IsaacZEncoder \
  --z-encoder-checkpoint $ISAAC_PROJECT/checkpoints/isaac_big_encoder_adapter.pt \
  --num-agents 4 \
  --num-episodes 20 \
  --episode-steps 150 \
  --gpu 0 \
  --action-format real \
  --velocity-format real \
  --output $ISAAC_PROJECT/test_runs/external_z_tdmpc2_test_001
```

Isaac Lab 示例：

```bash
/path/to/IsaacLab/isaaclab.sh -p $TDMPC_BRIDGE/isaac_test_tdmpc2_external_z.py \
  --project-root $TDMPC_BRIDGE \
  --model-path $TDMPC_BRIDGE/checkpoints/tdmpc2_irsim_trained.pt \
  --isaac-project-root $ISAAC_PROJECT \
  --isaac-env-module isaac_env_adapter.random_nav_env \
  --isaac-env-class IsaacRandomNavEnv \
  --isaac-env-kwargs-json '{"num_envs":1,"num_agents":4,"headless":true}' \
  --z-encoder-module isaac_z_adapter.external_z_encoder \
  --z-encoder-class IsaacZEncoder \
  --z-encoder-checkpoint $ISAAC_PROJECT/checkpoints/isaac_big_encoder_adapter.pt \
  --num-agents 4 \
  --num-episodes 20 \
  --episode-steps 150 \
  --gpu 0 \
  --action-format real \
  --velocity-format real \
  --output $ISAAC_PROJECT/test_runs/external_z_tdmpc2_test_001
```

---

# Part D：重要参数说明

## 18. TD-MPC2 相关参数

| 参数 | 含义 |
|---|---|
| `--project-root` | 指向包含 `exp_config.py`、`tdmpc2.py`、`tdmpc2_common/`、`tools/` 的目录 |
| `--model-path` | 原 IRSim 工程训练好的 TD-MPC2 checkpoint |
| `--gpu` | 使用的 CUDA GPU id |
| `--tdmpc-cfg-args` | 传给 `exp_config.get_config()` 的额外参数 |
| `--cfg-overrides-json` | 读取 config 后再覆盖的参数，JSON dict |

示例：

```bash
--cfg-overrides-json '{"eval_mode":true,"tb_log_figures":false}'
```

---

## 19. Isaac env 相关参数

| 参数 | 含义 |
|---|---|
| `--isaac-project-root` | 可选，加入 `sys.path`，方便 import Isaac 工程模块 |
| `--isaac-env-module` | Isaac env 所在 Python module |
| `--isaac-env-class` | Isaac env 类名 |
| `--isaac-env-kwargs-json` | 传给 Isaac env 构造函数的 JSON 参数 |

示例：

```bash
--isaac-env-module isaac_env_adapter.random_nav_env
--isaac-env-class IsaacRandomNavEnv
--isaac-env-kwargs-json '{"num_envs":1,"num_agents":4,"headless":true}'
```

---

## 20. 外部 z encoder 相关参数

| 参数 | 含义 |
|---|---|
| `--z-encoder-module` | 外部 z encoder 所在 Python module |
| `--z-encoder-function` | 函数式接口名称，默认 `encode_z` |
| `--z-encoder-class` | 类式接口名称。如果提供，则优先使用类接口 |
| `--z-encoder-checkpoint` | 外部 encoder / adapter 的 checkpoint |
| `--z-encoder-kwargs-json` | 传给外部 z encoder 的额外 JSON 参数 |

示例：

```bash
--z-encoder-kwargs-json '{"use_amp":false,"map_key":"laser_map"}'
```

脚本会把这些额外参数传给类构造函数。函数式接口目前主要会收到：

```python
obs
agent_id
device
cfg
tdmpc_cfg
```

如果函数签名不包含某些参数，脚本会自动过滤。

---

## 21. 动作和速度参数

| 参数 | 推荐值 | 含义 |
|---|---|---|
| `--action-format` | `real` | 传给 Isaac env 的动作格式，`real` 表示 `[m/s, rad/s]` |
| `--velocity-key` | `velocity` | 从 obs 中读取速度的字段名 |
| `--velocity-format` | `real` | obs 中速度的格式，推荐真实速度 |

动作格式说明：

```text
--action-format real
    env.step(action) 收到 [v_mps, w_radps]

--action-format norm
    env.step(action) 收到 [v_norm, w_norm]，范围 [-1,1]
```

速度格式说明：

```text
--velocity-format real
    obs["velocity"] = [v_mps, w_radps]

--velocity-format model_norm
    obs["velocity"] = [v/max_linear_vel, (w/max_angular_vel+1)/2]

--velocity-format action_norm
    obs["velocity"] = [v_norm, w_norm]，范围 [-1,1]
```

---

## 22. 动态地图 / risk-aware planner 参数

| 参数 | 含义 |
|---|---|
| `--map-key` | 从 obs 中读取局部地图的字段名，默认 `laser_map` |
| `--map-normalize` | 地图归一化方式，默认 `auto` |
| `--map-invert` | 地图黑白语义反转 |
| `--disable-env-dyn-weight` | 关闭 TD-MPC2 planner 中基于环境动态地图的权重 |

重要说明：

测试时的 `z` 来自外部 encoder，不来自 `laser_map` 的内部处理。

但是，如果原 TD-MPC2 checkpoint 打开了 risk-aware dynamic-map 相关逻辑，例如：

```python
cfg.use_env_dyn_weight = True
```

那么 planner 可能需要从 observation 中构造动态区域图。由于现在没有原始 180 维 scan observation，测试脚本会用 Isaac obs 里的 `laser_map` 历史帧来构造一个 `[T, H, W]` 的 map history。

如果不想用这部分逻辑，请直接加：

```bash
--disable-env-dyn-weight
```

---

## 23. 输出参数

| 参数 | 含义 |
|---|---|
| `--output` | 保存测试结果的目录 |
| `--save-rollout` | 保存每一步的 z/action/reward/done 到 `rollout.pt` |
| `--print-every` | 每隔多少步打印一次日志 |

输出目录示例：

```text
external_z_tdmpc2_test_001/
├── test_meta.json
├── episode_metrics.csv
├── summary.json
└── rollout.pt
```

其中：

```text
test_meta.json
    本次测试使用的模型路径、env module、z encoder module、参数等。

episode_metrics.csv
    每个 episode 的步数、reward、success_count、collision_count、done_count。

summary.json
    所有 episode 的总体成功率、碰撞率、reward 总和等。

rollout.pt
    如果指定 --save-rollout，会保存每一步的 z、动作、reward、done、success、collision。
```

`rollout.pt` 中每条 record 大致包含：

```python
{
    "episode": int,
    "step": int,
    "agent": int,
    "z": torch.Tensor,
    "action_norm": torch.Tensor,
    "action_real": torch.Tensor,
    "new_traj": bool,
    "reward": float,
    "done": bool,
    "success": bool,
    "collision": bool,
}
```

---

# Part E：最小可运行骨架

## 24. Isaac env adapter 最小骨架

文件：

```text
<Isaac工程>/isaac_env_adapter/random_nav_env.py
```

示例：

```python
import numpy as np


class IsaacRandomNavEnv:
    def __init__(self, num_envs=1, num_agents=4, headless=True, **kwargs):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.headless = headless

        # TODO: 初始化 Isaac Sim / Isaac Lab 环境
        # self.sim = ...
        # self.robots = ...

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # TODO:
        # 1. reset Isaac world
        # 2. 随机初始化机器人位姿
        # 3. 随机初始化目标点
        # 4. 推进仿真，确保传感器可读

        return self._get_obs_all()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # 如果运行脚本使用 --action-format real:
        # action[i] = [v_mps, w_radps]

        # TODO:
        # for i in range(self.num_agents):
        #     v, w = action[i]
        #     self.set_robot_velocity(i, v, w)

        # TODO: 推进 Isaac 仿真一步
        # self.sim.step()

        obs = self._get_obs_all()
        reward = np.zeros(self.num_agents, dtype=np.float32)
        done = np.zeros(self.num_agents, dtype=bool)
        info = {
            "success": np.zeros(self.num_agents, dtype=bool),
            "collision": np.zeros(self.num_agents, dtype=bool),
        }

        return obs, reward, done, info

    def _get_obs_all(self):
        return [self._get_obs_one(i) for i in range(self.num_agents)]

    def _get_obs_one(self, i):
        laser_map = self._build_laser_map(i)
        velocity = self._get_robot_velocity(i)
        goal_rel = self._get_goal_relative_polar(i)

        return {
            "laser_map": laser_map.astype(np.float32),
            "velocity": np.asarray(velocity, dtype=np.float32),
            "goal_rel": np.asarray(goal_rel, dtype=np.float32),
        }

    def _build_laser_map(self, i):
        # TODO: 返回当前机器人局部地图 [128,256]
        return np.ones((128, 256), dtype=np.float32)

    def _get_robot_velocity(self, i):
        # TODO: 返回当前机器人真实速度 [m/s, rad/s]
        return np.array([0.0, 0.0], dtype=np.float32)

    def _get_goal_relative_polar(self, i):
        # TODO: 返回机器人局部坐标系下目标 [distance, theta]
        return np.array([3.0, 0.0], dtype=np.float32)

    def close(self):
        # TODO: 如有需要，关闭 Isaac app / sim
        pass
```

---

## 25. 外部 z encoder 最小骨架

文件：

```text
<Isaac工程>/isaac_z_adapter/external_z_encoder.py
```

示例：

```python
import torch


class IsaacZEncoder:
    def __init__(self, checkpoint_path=None, device="cuda", cfg=None, **kwargs):
        self.device = torch.device(device)
        self.cfg = cfg

        # TODO: 创建并加载你们的大 encoder + adapter
        # self.encoder = ...
        # if checkpoint_path is not None:
        #     ckpt = torch.load(checkpoint_path, map_location=self.device)
        #     self.encoder.load_state_dict(ckpt["model"])
        # self.encoder.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, obs, agent_id=None, device=None, cfg=None):
        # TODO: 这里替换成真实模型推理
        # laser_map = obs["laser_map"]
        # velocity = obs["velocity"]
        # goal_rel = obs["goal_rel"]
        # z = self.encoder(laser_map, velocity, goal_rel)

        z_dim = int(self.cfg.latent_dim) * int(getattr(self.cfg, "enc_num", 1))
        z = torch.zeros(1, z_dim, device=self.device, dtype=torch.float32)
        return z
```

---

# Part F：自检和常见问题

## 26. z encoder 接口自检

在正式跑完整测试前，建议先让 Isaac 同学单独测试：

```python
obs = env.reset()[0] if isinstance(env.reset(), list) else env.reset()
z = encoder(obs, agent_id=0, device="cuda", cfg=cfg)
print(type(z))
print(z.shape)
print(torch.isfinite(z).all() if isinstance(z, torch.Tensor) else np.isfinite(z).all())
```

期望：

```text
shape = [Z] 或 [1, Z]
无 NaN / Inf
Z = cfg.latent_dim * cfg.enc_num
```

---

## 27. Isaac obs 自检

建议在 `reset()` 后打印：

```python
obs = env.reset()
obs0 = obs[0] if isinstance(obs, list) else obs

print(obs0.keys())
print(obs0["laser_map"].shape, obs0["laser_map"].dtype, obs0["laser_map"].min(), obs0["laser_map"].max())
print(obs0["velocity"])
print(obs0["goal_rel"])
```

推荐输出：

```text
laser_map shape = (128, 256)
laser_map value range = [0, 1] 或 [0, 255]
velocity = [v_mps, w_radps]
goal_rel = [distance, theta]
```

---

## 28. 常见问题

### 问题 1：`ModuleNotFoundError: No module named 'isaac_env_adapter'`

说明 `PYTHONPATH` 没有包含 Isaac 工程根目录。

解决：

```bash
export PYTHONPATH=$ISAAC_PROJECT:$TDMPC_BRIDGE:$PYTHONPATH
```

或者使用：

```bash
--isaac-project-root $ISAAC_PROJECT
```

---

### 问题 2：`ModuleNotFoundError: No module named 'tdmpc2_common'`

说明 `PYTHONPATH` 没有包含 `tdmpc_bridge`，或者没有复制完整 `tdmpc2_common/`。

检查：

```bash
ls $TDMPC_BRIDGE/tdmpc2_common
ls $TDMPC_BRIDGE/tools
```

---

### 问题 3：`z dim mismatch`

脚本打印类似：

```text
[WARN] z dim mismatch: got xxx, expected cfg.latent_dim*cfg.enc_num=yyy
```

原因：外部 encoder + adapter 输出维度和原 TD-MPC2 checkpoint 后半部分不一致。

需要检查：

```python
cfg.latent_dim
cfg.enc_num
```

以及 adapter 最后一层输出维度。

---

### 问题 4：动作发给 Isaac 后机器人不动

检查：

```bash
--action-format real
```

此时 Isaac 收到的是：

```python
[v_mps, w_radps]
```

如果 Isaac 环境实际期望 `[-1,1]` 归一化动作，应改成：

```bash
--action-format norm
```

还要检查 Isaac env 里是否真的把 action 发给了机器人控制接口。

---

### 问题 5：轨迹很抖或速度约束异常

重点检查 `obs["velocity"]` 和 `--velocity-format` 是否一致。

如果 `obs["velocity"]` 是真实速度：

```bash
--velocity-format real
```

如果 `obs["velocity"]` 已经是 `[0,1]` 的模型归一化速度：

```bash
--velocity-format model_norm
```

如果 `obs["velocity"]` 是 `[-1,1]` 动作归一化格式：

```bash
--velocity-format action_norm
```

---

### 问题 6：risk-aware planner 报和 observation / CostMap 有关的错

当前测试不再使用原 180 维 scan observation。如果原 checkpoint 的 risk-aware dynamic map 逻辑需要旧 observation，可以先关闭：

```bash
--disable-env-dyn-weight
```

如果需要保留该逻辑，确保 Isaac obs 中存在：

```python
"laser_map": np.ndarray [128,256]
```

脚本会维护历史 map，并 patch planner 的动态地图构造函数。

---

### 问题 7：CUDA 相关报错

原 TD-MPC2 代码中有硬编码 CUDA。建议使用有 GPU 的 Isaac 环境运行。

运行时指定：

```bash
--gpu 0
```

---

# 29. 对接时最重要的三句话

1. **Isaac env 每个 step 返回每个 agent 的 obs；obs 至少包含 `laser_map`、`velocity`、`goal_rel`。**

2. **外部 z encoder 接口输入单个 agent 的 obs，输出 shape 为 `[Z]` 或 `[1,Z]` 的隐变量 z。**

3. **测试脚本会加载原 TD-MPC2 checkpoint，但只使用 checkpoint 中 encoder 之后的 dynamics / reward / Q / policy / planner，不再使用原 encoder。**
