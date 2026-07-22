# Genie-Envisioner Web Inference 客户端构建说明

本文说明如何基于仓库已有的 `web_infer_scripts` 和 `web_infer_utils` 构建 Isaac Sim 客户端。当前代码已经提供 websocket 服务端和 Python 客户端通信工具，因此不需要重写一套网络脚本；客户端需要做的是按服务端约定封装 Isaac Sim 中采集到的图像、机器人状态、任务文本和执行步长，并把服务端返回的动作序列送回仿真控制器执行。

## 结论

`web_infer` 天然支持“另一台服务器上的客户端通过网口请求本机推理服务端”的模式。

已有能力：

- 服务端入口：`web_infer_scripts/main_server.py`
- 服务端实现：`web_infer_utils/server.py`
- 推理执行器：`web_infer_utils/MVActor.py`
- 客户端 websocket 封装：`web_infer_utils/openpi_client/websocket_client_policy.py`
- 随机观测压测示例：`web_infer_scripts/simple_client.py`

仍需客户端侧实现：

- 从 Isaac Sim 采集多视角 RGB 图像，并整理成服务端要求的 `obs`。
- 从 Isaac Sim / robot articulation 读取当前机器人状态，并整理成服务端要求的 `state`。
- 按仿真控制频率、通信耗时、动作执行耗时计算 `execution_step`。
- 将返回的 `actions` 拆成每个仿真 tick 的机器人控制命令。
- 处理连接中断、超时、服务端报错、任务 reset。

## 通信拓扑

推荐部署方式：

```text
Isaac Sim 客户端服务器                         推理服务端服务器
----------------------                         ----------------
采集相机图像 / 机器人状态
构造 msgpack-numpy 请求       websocket        main_server.py
WebsocketClientPolicy  --------------------->  MVActorServer
执行返回动作                 actions          MVActor.play()
```

默认协议是 websocket，默认端口是 `8001`，消息序列化使用 `msgpack_numpy`，可以直接传输 `numpy.ndarray`。服务端还暴露 `GET /healthz` 健康检查，返回 `OK`。

## 服务端启动

在推理服务端，即当前 Genie-Envisioner 所在机器上启动：

```bash
cd /mnt/pfs/s7fsio/code/pyl/tau-0/Genie-Envisioner

python3 web_infer_scripts/main_server.py \
  -c configs/ltx_model/policy_model.yaml \
  -w /path/to/checkpoint_or_transformer_dir \
  --host 0.0.0.0 \
  --port 8001 \
  --domain_name DATASETNAME \
  --denoise_step 10 \
  --action_dim 14 \
  --add_state
```

参数含义：

- `--host`：服务端监听地址。跨机器访问时建议设为 `0.0.0.0` 或服务端网卡 IP；`localhost` 只能本机访问。
- `--port`：websocket 端口，默认 `8001`。
- `-c / --config`：训练/推理 YAML 配置。该配置决定 `action_type`、`action_space`、`action_chunk`、图像尺寸、模型类和 pipeline 类。
- `-w / --weight`：训练好的 diffusion transformer 权重目录或权重路径，传给 `load_diffusion_model(model_dir=...)`。
- `--domain_name`：统计量 key 的前缀。服务端会读取 `<domain_name>_state_<action_space>` 和动作统计量 key。
- `--denoise_step`：单次推理 denoise steps。越大通常越慢。
- `--action_dim`：服务端输出动作维度。必须和训练配置、状态维度、控制器期望一致。
- `--add_state`：该参数目前传入了入口脚本，但 `MVActor` 实际读取的是 YAML 里的 `add_state` 字段；如果模型需要 state conditioning，请确保 YAML 中也有 `add_state: true`。

网络侧需要确认：

- 客户端能访问 `ws://<server_ip>:8001`。
- 防火墙或容器端口映射已放通 `8001`。
- 如果有多网卡，`--host` 使用客户端能访问的网卡 IP，或使用 `0.0.0.0` 监听全部网卡。

## 客户端依赖

Isaac Sim 客户端至少需要：

```bash
pip install websockets==14.1 msgpack-python numpy typing_extensions
```

如果直接复用仓库中的 `WebsocketClientPolicy`，客户端需要能 import 以下文件：

```text
web_infer_utils/openpi_client/websocket_client_policy.py
web_infer_utils/openpi_client/msgpack_numpy.py
web_infer_utils/openpi_client/base_policy.py
```

推荐做法是在 Isaac Sim 客户端工程中保留同样的包结构，或者把 `Genie-Envisioner/web_infer_utils` 加到 `PYTHONPATH`。

## 请求输入 Schema

客户端每次调用服务端时发送一个 Python `dict`：

```python
message = {
    "obs": images,
    "state": robot_state,
    "prompt": prompt,
    "execution_step": execution_step,
}
```

### `obs`

类型：

- `numpy.ndarray`

允许形状：

- 推荐：`(V, H, W, 3)`，`dtype=np.uint8`，取值 `[0, 255]`
- 也支持：`(V, 3, H, W)`，浮点型，取值 `[-1, 1]`

变量含义：

- `V`：相机视角数量。示例客户端用 `3`。
- `H, W`：图像高度和宽度。必须和训练配置的 `data.train.sample_size` / 训练预处理一致；示例为 `192 x 256`。
- 通道顺序：RGB。Isaac Sim 可能给 RGBA 或 BGR，需要转换成 RGB，并去掉 alpha。

服务端逻辑：

- 如果 `obs.dtype == np.uint8`，服务端会做 `obs / 127.5 - 1`，然后从 `(V,H,W,3)` 转成 `(V,3,H,W)`。
- 服务端会维护 `n_prev=4` 个历史观测，用于构造模型输入。

### `state`

类型：

- `numpy.ndarray`

推荐形状：

- `(action_dim,)`

变量含义：

- 当前机器人状态，通常是当前关节位置、末端位姿或训练时定义的状态向量。
- 维度和顺序必须与训练数据的 `state` 定义一致。

服务端逻辑：

- 如果 YAML 中 `add_state: true`，服务端会用统计量归一化 `state`，作为 `history_action_state` 输入模型。
- 无论是否 `add_state`，当前 `MVActor.play()` 后处理动作时都会使用 `state`，尤其 `action_type == "delta"` 或 `"relative"` 时会把预测量转换回绝对控制量。
- 因此客户端应始终发送合法 `state`，不要传 `None`。

### `prompt`

类型：

- `str`

变量含义：

- 当前任务文本描述。
- 如果字符串中包含 `<reset>`，服务端会先调用 `MVActor.reset()` 清空历史图像和动作 buffer，然后移除 `<reset>` 再执行推理。

推荐用法：

- 新 episode 的第一帧：`"<reset>" + task_instruction`
- 同一 episode 后续帧：`task_instruction`

### `execution_step`

类型：

- `int`

范围：

- 当前代码断言 `1 <= execution_step <= 100`

变量含义：

- 从上一次请求到这一次请求之间，客户端实际已经执行了多少个控制步。
- 服务端会返回 `execution_step` 条动作，即输出 `actions.shape[0] == execution_step`。

服务端逻辑：

- 服务端内部 `count += execution_step`。
- 当 `count >= threshold` 时，服务端将当前观测写入历史窗口；否则只刷新最近观测。
- 这使客户端可以把网络延迟、服务端推理耗时、动作执行时间折算为实际执行步数，避免服务端历史记忆和仿真进度明显错位。

### 可选字段

`MVActor.play()` 还支持以下可选字段：

- `idx`：默认 `0`，当前代码未用于核心推理。
- `num_inference_steps`：覆盖服务端默认 denoise steps。
- `state_zeropadding`：形如 `[left_pad, right_pad]`，用于 state 归一化前后补零。
- `ndim_action`：限制输出动作维度，默认等于 `action_dim`。

## 响应输出 Schema

服务端返回：

```python
response = {
    "actions": actions,
}
```

### `actions`

类型：

- `numpy.ndarray`

形状：

- `(execution_step, ndim_action)`

变量含义：

- 每一行是一个控制步的动作。
- 动作已经在服务端按训练统计量反归一化。
- 如果训练配置是 `action_type: delta`，服务端已经基于当前 `state` 累加成目标动作。
- 如果训练配置是 `action_type: relative`，服务端已经把相对量加回状态并反归一化。
- 如果训练配置是 `action_type: absolute`，服务端直接输出反归一化后的绝对动作。

客户端执行逻辑：

- 对于 joint action：按训练约定拆成左臂、夹爪、右臂、夹爪等字段，再写入 Isaac Sim articulation controller。
- 对于末端位姿 action：按训练约定做 IK 或控制器转换。
- 动作频率必须和训练数据采样频率一致，或用插值/重复策略对齐。

## Isaac Sim 客户端模块建议

建议把客户端拆成以下模块：

```text
isaac_client/
  config.py              # 服务端地址、端口、相机名、动作维度、控制频率
  observation_builder.py # 从 Isaac Sim 采集 obs/state/prompt
  ge_policy_client.py    # websocket 连接、重连、infer 封装
  latency_controller.py  # 根据耗时估计 execution_step
  action_executor.py     # 将 actions 写入 Isaac Sim 控制器
  main_loop.py           # 仿真主循环
```

### `ObservationBuilder`

职责：

- 从 Isaac Sim 相机读取 RGB。
- 将每个视角 resize 到训练尺寸。
- 保证输出为 `np.uint8`、RGB、shape `(V,H,W,3)`。
- 从机器人读取当前状态，整理为 `(action_dim,)`。
- 在 episode reset 时给 prompt 加 `<reset>`。

核心函数建议：

```python
class ObservationBuilder:
    def build(self, task_instruction: str, reset: bool, execution_step: int) -> dict:
        images = self.read_images()
        state = self.read_robot_state()
        prompt = f"<reset>{task_instruction}" if reset else task_instruction
        return {
            "obs": images,
            "state": state,
            "prompt": prompt,
            "execution_step": execution_step,
        }
```

关键变量：

- `camera_names`：相机顺序，必须和训练视角顺序一致。
- `image_size`：训练图像尺寸，如 `(192, 256)`。
- `state_keys`：state 维度顺序定义。
- `action_dim`：动作/状态维度，必须和服务端一致。

### `GEPolicyClient`

职责：

- 建立 websocket 连接。
- 发送 `dict` 请求。
- 接收 `actions`。
- 处理服务端字符串错误和连接中断。

可直接复用：

```python
from web_infer_utils.openpi_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host=SERVER_IP, port=8001)
metadata = policy.get_server_metadata()
response = policy.infer(message)
actions = response["actions"]
```

建议增强逻辑：

- `connect_timeout`：避免 Isaac Sim 主循环无限阻塞。
- `infer_timeout`：一次推理超过阈值时，继续执行上一轮剩余动作或进入安全停止。
- `reconnect()`：服务端重启后自动重连。
- `last_actions`：网络失败时缓存上一轮动作，短时间容错。

### `LatencyController`

职责：

- 根据仿真控制频率和一次请求耗时，估算下一次请求的 `execution_step`。
- 限制 `execution_step` 在 `[1, 100]`。
- 避免网络抖动导致动作 chunk 忽长忽短。

推荐逻辑：

```python
class LatencyController:
    def __init__(self, control_hz: float, min_step: int = 1, max_step: int = 100):
        self.control_dt = 1.0 / control_hz
        self.min_step = min_step
        self.max_step = max_step

    def estimate_execution_step(self, elapsed_sec: float) -> int:
        steps = round(elapsed_sec / self.control_dt)
        return int(np.clip(steps, self.min_step, self.max_step))
```

实际闭环建议：

- 首帧 `execution_step=1`，并在 prompt 加 `<reset>`。
- 记录一次 request 的开始时间 `t0`。
- 收到响应并执行完返回动作后，计算 `elapsed_sec = time.monotonic() - t0`。
- 下一次请求使用 `round(elapsed_sec * control_hz)` 作为 `execution_step`。
- 如果你固定每次只执行 `K` 个控制步再请求下一轮，也可以直接设 `execution_step=K`。

### `ActionExecutor`

职责：

- 将 `actions[t]` 转换成 Isaac Sim 控制命令。
- 每个仿真 tick 执行一行动作。
- 做限幅、速度限制、安全停止。

核心函数建议：

```python
class ActionExecutor:
    def execute_chunk(self, actions: np.ndarray) -> int:
        executed = 0
        for action in actions:
            command = self.convert_action(action)
            self.apply_command(command)
            self.step_sim()
            executed += 1
        return executed
```

关键变量：

- `joint_order`：动作向量中的关节顺序。
- `gripper_indices`：夹爪维度位置。
- `action_type`：客户端不需要再做 delta/relative 反归一化，但需要知道动作表示是 joint target、eef target 还是其他控制接口。
- `control_hz`：动作执行频率。
- `safety_limits`：关节范围、速度范围、夹爪范围。

## 主循环示例

```python
import time
import numpy as np

from web_infer_utils.openpi_client.websocket_client_policy import WebsocketClientPolicy


SERVER_IP = "10.0.0.12"
PORT = 8001
CONTROL_HZ = 20


policy = WebsocketClientPolicy(host=SERVER_IP, port=PORT)
print("server metadata:", policy.get_server_metadata())

execution_step = 1
reset = True
task_instruction = "Pick up the object and place it into the box."

while simulation_is_running():
    t0 = time.monotonic()

    message = observation_builder.build(
        task_instruction=task_instruction,
        reset=reset,
        execution_step=execution_step,
    )
    reset = False

    try:
        response = policy.infer(message)
        actions = response["actions"]
    except Exception as exc:
        actions = safety_policy.get_fallback_actions()
        logger.exception("GE server request failed: %s", exc)

    executed_steps = action_executor.execute_chunk(actions)
    elapsed_sec = time.monotonic() - t0

    measured_steps = round(elapsed_sec * CONTROL_HZ)
    execution_step = int(np.clip(max(executed_steps, measured_steps), 1, 100))
```

## 延迟处理建议

网络推理闭环里会出现三类延迟：

- 图像采集和 resize 延迟。
- websocket 传输和 msgpack 序列化延迟。
- 服务端模型推理延迟。

建议策略：

- 客户端不要每个 Isaac Sim tick 都请求服务端；每轮请求返回一个 action chunk，然后连续执行这个 chunk。
- `execution_step` 使用“实际执行步数”或“总耗时折算步数”，让服务端历史观测更新频率贴近真实仿真进度。
- 如果服务端一次推理慢于控制周期，客户端继续执行上一轮剩余动作；没有剩余动作时执行安全停止或低速保持。
- 如果通信抖动大，可以把 `execution_step` 固定为较小 chunk，例如 `4` 或 `8`，并在客户端异步预取下一轮动作。
- 如果使用异步预取，务必保证发送给服务端的 `state` 和 `obs` 是当前最新状态，而不是上一次请求时的旧状态。

## 可用性检查

服务端启动后先做健康检查：

```bash
curl http://<server_ip>:8001/healthz
```

期望输出：

```text
OK
```

然后在客户端机器上运行随机观测压测：

```bash
cd /path/to/Genie-Envisioner
python3 web_infer_scripts/simple_client.py --host <server_ip> --port 8001 --env WM
```

如果随机压测通过，说明网络、websocket、msgpack-numpy 和服务端推理入口是通的。之后再接入 Isaac Sim 的真实 `obs/state`。

## 常见问题

### 客户端连不上服务端

检查：

- 服务端是否用了 `--host localhost`。跨机器时应使用 `0.0.0.0` 或真实网卡 IP。
- 端口是否被防火墙拦截。
- 客户端访问的是不是服务端内网 IP。
- `curl http://<server_ip>:8001/healthz` 是否返回 `OK`。

### 服务端报统计量 KeyError

检查：

- `--domain_name` 是否和统计文件里的 key 前缀一致。
- YAML 中 `data.train.action_space` 是否和统计文件后缀一致。
- `action_type == "delta"` 时动作统计 key 是 `<domain_name>_delta_<action_space>`。
- state 统计 key 是 `<domain_name>_state_<action_space>`。

### 动作维度不一致

检查：

- 服务端 `--action_dim`。
- YAML 中 `diffusion_model.config.action_in_channels`。
- 训练统计文件中的动作均值/方差维度。
- Isaac Sim 客户端读取的 `state.shape` 和执行器期望动作维度。

### 图像 shape 不一致

检查：

- 客户端是否传了 `(V,H,W,3)` 的 RGB `uint8`。
- 是否误传了 RGBA，需要去掉 alpha。
- 是否误传了 BGR，需要转成 RGB。
- `V` 的视角数和训练时一致。
- `H,W` 和训练 resize 尺寸一致。

### reset 后动作异常

新 episode 第一帧 prompt 必须包含 `<reset>`，否则服务端会沿用上一轮历史观测。reset 只需要放在第一帧，后续请求使用普通 prompt。

## 需要改函数时的边界

一般情况下不需要改服务端代码。只有以下情况才建议改：

- 需要返回更多信息，比如模型置信度、预测视频、服务端耗时。
- 需要鉴权或 TLS。
- 需要异步多客户端并发。
- 需要改变动作后处理逻辑。

建议改动位置：

- 改请求/响应字段：`web_infer_utils/server.py` 的 `_handler()`。
- 改模型输入、归一化、历史帧逻辑：`web_infer_utils/MVActor.py` 的 `play()`。
- 改服务端启动参数：`web_infer_scripts/main_server.py` 的 `get_args()` 和 `MVActorServer(...)` 初始化。
- 改客户端通信逻辑：`web_infer_utils/openpi_client/websocket_client_policy.py`。

不要在 Isaac Sim 客户端重复实现服务端已有的归一化和动作反归一化逻辑；客户端只需要保持输入数据顺序、维度、单位与训练数据一致。
