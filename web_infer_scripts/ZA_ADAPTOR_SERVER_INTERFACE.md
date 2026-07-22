# Za Adaptor Server Interface

This server follows the same websocket/msgpack-numpy transport style as
`web_infer_scripts/main_server.py`, but its internal policy is:

```text
client obs + beta
-> frozen LTX or checkpoint-selected WorldVLN video backbone
-> Zb
-> beta fusion + ZaAdaptor
-> external z / Za'
-> TD-MPC2 planner from isaac_test_tdmpc2_external_z.py
-> robot action
```

The client receives actions, not raw `z`. `z` can be returned only as optional
debug output.

### Buffered inference and visual cache

The server advances its rolling RGB history on every request. By default it
recomputes WorldVLN from the latest eight-frame window every time, including
requests where TD-MPC returns a buffered action (`new_traj=False`). Responses
expose `visual_cache_hit`, which remains false in this mode.

The previous trajectory-level optimization remains available as an explicit
opt-in:

```yaml
server:
  reuse_visual_zb_for_buffered_actions: true
```

When enabled, the server runs the frozen video backbone only when TD-MPC needs
a new trajectory. While the action buffer is non-empty it reuses the visual
`Zb`, while still recomputing lightweight beta fusion and ZaAdaptor.

This is a trajectory-level exact cache. InfinityStar's VideoVAE encodes a
temporal clip and its visual Transformer uses non-causal attention, so replacing
one frame changes the representations of the other seven frames. A literal
"encode only the newest frame" KV cache would not reproduce the feature
distribution used to train this checkpoint. The full latest eight-frame window
is therefore refreshed at each new trajectory boundary.

## Start Server

```bash
cd /home/pengyulin/h20_2_code/pyl/tau-0/Genie-Envisioner

TDMPC_PROJECT_ROOT=/path/to/tdmpc_bridge \
TDMPC_MODEL_PATH=/path/to/tdmpc_checkpoint.pt \
bash web_infer_scripts/run_za_adaptor_server.sh
```

`TDMPC_PROJECT_ROOT` must point to the directory that contains
`exp_config.py`, `tdmpc2.py`, `tdmpc2_common/`, and `tools/`.
`TDMPC_MODEL_PATH` is the trained TD-MPC2 checkpoint loaded by
`isaac_test_tdmpc2_external_z.py`.

Equivalent explicit command:

```bash
python3 web_infer_scripts/main_za_adaptor_server.py \
  -c configs/ltx_model/za_adaptor_tdmpc_server.yaml \
  -w OUTPUTS/ltx/za_adaptor_tdmpc/2026_07_09_17_38_25/step_20000/za_adaptor.pt \
  --tdmpc_project_root /path/to/tdmpc_bridge \
  --tdmpc_model_path /path/to/tdmpc_checkpoint.pt \
  --host 0.0.0.0 \
  --port 8011 \
  --device cuda \
  --dtype bf16 \
  --action_format real \
  --disable_online_training
```

For WorldVLN checkpoints, the server reads `worldvln_video_backbone` from
`za_adaptor.pt`, restores `worldvln_feature_projector_state_dict`, and skips
loading the LTX backbone. In `feature_mode: intermediate`, requests must contain
the single camera view used by training and prompt text is not used by the
visual feature extractor (the `<reset>` marker still resets history).

Health check:

```bash
curl http://<server_ip>:8011/healthz
```

The first websocket message from the server is metadata. After that, each client
request gets one response.

## Request Schema

Send a Python `dict` through `WebsocketClientPolicy.infer(...)` or any compatible
msgpack-numpy websocket client.

```python
message = {
    "obs": images,
    "prompt": "<reset>navigation task",
    "execution_step": 1,

    # Option A: send beta directly
    "beta": np.array([vx, vy, vyaw, goal_r, goal_theta], dtype=np.float32),

    # Option B: send state and goal separately instead of beta
    # "robot_state": np.array([vx, vy, vyaw], dtype=np.float32),
    # "goal_rel": np.array([goal_r, goal_theta], dtype=np.float32),

    # Optional: planner map for TD-MPC2 risk-aware dynamic-map logic.
    # "laser_map": laser_map,

    # Optional: online TD-MPC training feedback for the previous server action.
    # Send these on the next control step after executing response["action_norm"]
    # or response["actions"] in the environment.
    # "reward": float_reward_from_previous_step,
    # "done": bool_episode_done,
}
```

### `obs`

Same as the existing GE web inference server:

```text
shape: [V, H, W, 3]
dtype: uint8
range: [0, 255]
```

or:

```text
shape: [V, 3, H, W]
dtype: float32/float16
range: [-1, 1]
```

The default config expects `H=192`, `W=256`, and keeps `n_prev=8` frames in the
server-side history buffer.

### `prompt`

String prompt passed to the frozen video model text encoder. If it contains
`<reset>`, the server clears its image history before inference.

For navigation tests, a generic prompt is acceptable if the model was trained
with unified captions:

```python
prompt = "<reset>best quality, consistent and smooth motion, realistic, clear and distinct."
```

### `execution_step`

Integer. Used only to update the image-memory buffer, matching the existing
server behavior. Use `1` for one request per control step.

### `beta`

Preferred direct interface:

```text
beta = [vx, vy, vyaw, goal_r, goal_theta]  # beta_dim=5 checkpoints
```

For the current `beta_dim=10` 8D-state checkpoint, send:

```text
beta = [torso_r, torso_p, torso_y, hb, pyaw, vx, vy, vyaw, goal_r, goal_theta]
```

Field meaning:

```text
vx         : robot x/forward velocity, same convention as training data
vy         : robot y/lateral velocity, same convention as training data
vyaw       : angular velocity / yaw-rate
goal_r     : target distance in robot-centric polar coordinates
goal_theta : target angle in robot-centric polar coordinates
```

This matches the training converter's `--state_mode 3d` beta layout.

### `robot_state` + `goal_rel`

Alternative interface when the client does not want to concatenate beta:

```python
"robot_state": np.array([vx, vy, vyaw], dtype=np.float32)
"goal_rel": np.array([goal_r, goal_theta], dtype=np.float32)
```

The server will concatenate them into:

```python
beta = [vx, vy, vyaw, goal_r, goal_theta]
```

For TD-MPC2 planning constraints, the server maps this beta layout to:

```text
TD-MPC2 velocity = [vx, vyaw]
TD-MPC2 goal_rel = [goal_r, goal_theta]
```

If your client has a more exact current velocity convention for TD-MPC2, send it
explicitly:

```python
"velocity": np.array([linear_v, angular_w], dtype=np.float32)
```

Then `velocity` overrides the `[vx, vyaw]` value derived from beta.

### `laser_map` / `map`

Optional. If the loaded TD-MPC2 config uses environment dynamic-risk weighting,
the planner may need a map history. The server accepts any of these keys:

```text
laser_map, lidar_map, costmap, obs_map, map
```

The value follows the map conventions supported by
`PlannerMapHistory` in `isaac_test_tdmpc2_external_z.py`.

### Online TD-MPC Training Feedback

If `tdmpc.online_training.enabled: true`, the server builds transitions directly
from continuous client requests:

```text
previous z, previous action_norm, current z, reward, done
```

Therefore the client should send `reward` and `done` with the next request after
executing the previous action. Example control loop:

```python
request_t = {
    "obs": obs_t,
    "robot_state": state_t,
    "goal_rel": goal_rel_t,
    "reward": reward_from_action_t_minus_1,
    "done": done_from_action_t_minus_1,
}
response_t = policy.infer(request_t)
execute(response_t["action_norm"])  # or response_t["actions"], depending on action_format
```

If `reward` is absent, online training still uses latent dynamics consistency and
behavior cloning against the server action. Reward-head training is used only
when `tdmpc.online_training.reward_weight > 0` and the request contains
`reward`; Q training is used only when `tdmpc.online_training.q_weight > 0` and
the request contains `reward`.

If `tdmpc.online_training.train_adaptor: true`, the current request also keeps
the torch graph through:

```text
BetaZBFusion -> ZaAdaptor -> current z
```

The video model, VAE, and text encoder are still frozen. The online loss can then
update `BetaZBFusion` and `ZaAdaptor` through the `e2e_*` terms while replayed
historical transitions remain detached.

## Response Schema

```python
response = {
    "actions": np.ndarray,    # default action format is real [m/s, rad/s]
    "action": np.ndarray,     # alias of actions
    "action_norm": np.ndarray,# normalized TD-MPC2 action, usually [v_norm, w_norm]
    "action_real": np.ndarray,# real action, usually [linear_v, angular_w]
    "new_traj": bool,
    "server_timing": {
        "infer_action_ms": float,
    },
    # Present only when tdmpc.online_training.enabled is true.
    "online_training": {
        "buffer_size": int,
        "num_updates": int,
        "last_update": None | {
            "loss": float,
            "dynamics_loss": float,
            "bc_loss": float,
            "reward_loss": float,
            "q_loss": float,
            "e2e_dynamics_loss": float,
            "e2e_bc_loss": float,
            "e2e_reward_loss": float,
            "e2e_q_loss": float,
            "grad_norm": float,
        },
    },
}
```

`actions` is controlled by `tdmpc.action_format` in the yaml or by
`--action_format` / `ACTION_FORMAT` at launch:

```text
real : response["actions"] = action_real, [m/s, rad/s]
norm : response["actions"] = action_norm, normalized TD-MPC2 action
```

Internally the server computes:

```text
obs alpha -> frozen LTX video model -> Zb
beta -> beta fusion
Zb_beta -> ZaAdaptor -> z
z -> TD-MPC2 external-z controller -> action
```

If `tdmpc.return_z: true` is set in the yaml, the response also includes:

```python
"z": np.ndarray,   # shape [512]
"za": np.ndarray,  # alias of z
```

## Minimal Client Example

```python
import numpy as np
from web_infer_utils.openpi_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="SERVER_IP", port=8011)
print(policy.get_server_metadata())

request = {
    "obs": np.random.randint(0, 256, size=(1, 192, 256, 3), dtype=np.uint8),
    "prompt": "<reset>best quality, consistent and smooth motion, realistic, clear and distinct.",
    "execution_step": 1,
    "robot_state": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "goal_rel": np.array([3.0, 0.0], dtype=np.float32),
}

response = policy.infer(request)
action = response["actions"]
print(action.shape)  # [2] for the default TD-MPC2 action_dim=2
print(response["action_norm"], response["action_real"])
```

## Notes

- Port `8011` is used by default so this server can run alongside the action
  server on port `8001`.
- The checkpoint is `za_adaptor.pt`, not `diffusion_pytorch_model.safetensors`.
- The service must also load a TD-MPC2 checkpoint. Set `tdmpc.project_root` and
  `tdmpc.model_path` in the yaml, or pass `TDMPC_PROJECT_ROOT` and
  `TDMPC_MODEL_PATH` when launching.
- If the repo lives under `/mnt/pfs/...` instead of `/home/pengyulin/...`, update
  the absolute paths in `configs/ltx_model/za_adaptor_tdmpc_server.yaml`.
