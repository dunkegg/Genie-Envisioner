# Za Adaptor Training

## Goal

This branch trains only a small adaptor network:

```text
data -> frozen tau-0/LTX video DiT -> Zb -> ZaAdaptor -> Za'
TDMPC latent from dataset -------------------------------> Za
loss = MSE(Za', Za)
```

For TDMPC-style experts, `Za` may depend on an extra condition `beta`
(velocity plus goal polar information). In that case the trainer can use:

```text
data alpha -> frozen video DiT -> Zb
beta ---------------------------> BetaZBFusion(Zb, beta) -> Zb_beta
Zb_beta -> ZaAdaptor -> Za'
Za -----------------------------> MSE supervision
```

This corresponds to treating the video model feature as a prior-like
`pi_p(Zb | alpha)` and using `beta` to learn a likelihood correction toward
`pi(Zb | alpha, beta)`.

The TDMPC path is assumed to be precomputed. The dataset should expose one extra
per-frame latent entry, usually a path to `latent.pt`, through a parquet column.

## Variable Information

For the current LTX config:

- `Zb` is taken from the last video transformer block before `norm_out/proj_out`.
- `Zb` shape is `[B, V * T_latent * H_latent * W_latent, C]`.
- `C = num_attention_heads * attention_head_dim = 32 * 64 = 2048`.
- The adaptor pools tokens by mean by default, so its input is `[B, 2048]`.
- `Za` defaults to `[B, 512]`, controlled by `za_adaptor.output_dim`.

The model also supports target tensors shaped `[B, T, D]`; the trainer pools them
with `za_adaptor.target_pool` before MSE.

## Adaptor Sizes

Default dimensions are `2048 -> 512`.

| size | hidden dims | approximate params | when to use |
| --- | --- | ---: | --- |
| `small` | `[512]` | 1.32M | quick smoke tests or small data |
| `base` | `[1024, 1024]` | 3.68M | recommended default |
| `large` | `[2048, 2048, 1024]` | 11.02M | large data, stronger nonlinear mapping |

You can also set `za_adaptor.hidden_dims` in YAML to override these presets.

`BetaZBFusion` also supports size presets. The current `base` preset matches the
original hand-written fusion config.

| size | fusion dim | beta hidden | heads | layers | beta tokens | approximate params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `small` | 256 | 128 | 4 | 1 | 2 | 1.91M |
| `base` | 512 | 256 | 8 | 1 | 4 | 5.79M |
| `large` | 1024 | 512 | 16 | 2 | 8 | 33.61M |

## Files

- `models/latent_adaptor.py`: adaptor MLP and target pooling.
- `runner/za_adaptor_trainer.py`: training runner that freezes VAE, text encoder,
  and video DiT.
- `configs/ltx_model/za_adaptor_lerobot.yaml`: example config.
- `scripts/train_za_adaptor.sh`: torchrun wrapper.

## Dataset Contract

The current adaptor trainer reuses `CustomLeRobotDataset`, so raw
`dataset_tdmpc/episode_*.h5 + chunk_*.pt` cannot be fed directly. Convert it to
the repo's LeRobot-like layout first.

For the TDMPC paired data:

```bash
python h5_tdmpc_2_lerobot.py \
  --input_dir dataset_tdmpc \
  --output_dir lerobot_data/tdmpc_za \
  --domain_name tdmpc_za \
  --latent_key z \
  --state_mode 3d
```

For `camera_pos` with 3 columns, `--state_mode 3d` reads it as
`[linear_velocity_x, linear_velocity_y, angular_velocity_yaw]`. Use
`--state_mode 8d` only for the older 8-column `camera_pos` files.

This writes:

```text
lerobot_data/tdmpc_za/
  meta/
  data/
  videos/
  latents/
  statistics.json
```

The converter also writes a `beta` parquet column. With `--state_mode 3d`,
`beta = [linear_velocity_x, linear_velocity_y, angular_velocity_yaw, goal_r, goal_theta]`
when the TDMPC pt file contains `goal`; otherwise it falls back to velocity only.
If you converted the dataset before this column existed, rerun the converter and
delete the old dataset cache files under `OUTPUTS/dataset_meta_info_cache/`.

Set these fields in `data.train` and `data.val`:

```yaml
za_latent_key: "latent"
za_latent_index_mode: "action"
za_latent_tensor_key: null
beta_key: "beta"
beta_index_mode: "last"
```

For adaptor-only training, `za_latent_index_mode: "last"` is also valid. Joint
TD-MPC training uses `"action"` so the trainer can compute dynamics consistency
over a Za sequence.

`za_latent_key` is the parquet column that stores either:

- a tensor/list/array directly;
- a path to `latent.pt`;
- a dict saved by `torch.save`.

If the `.pt` file is a dict with a nonstandard tensor key, set
`za_latent_tensor_key`.

`za_latent_index_mode`:

- `last`: use the last supervised action/frame index as the target Za.
- `video`: load Za for the selected video frame indexes and pool it.
- `action`: load Za for all action indexes and pool it.

Relative latent paths are resolved against the parquet chunk directory, the
dataset `data` directory, and the domain root.

## Run

From the repo root:

```bash
bash scripts/train_za_adaptor.sh main.py configs/ltx_model/za_adaptor_tdmpc.yaml
```

`configs/ltx_model/za_adaptor_tdmpc.yaml` now defaults to:

```yaml
train_mode: 'za_adaptor_tdmpc_joint'
```

This freezes the VAE, text encoder, and the full video prediction model, and
optimizes:

- Za adaptor and optional beta fusion.
- TD-MPC model parameters loaded from `tdmpc.model_path`.

The main supervised loss is still `MSE(za_pred, za_target)`. If
`joint_training.tdmpc_bc_weight > 0`, the trainer also feeds `za_pred` into
`tdmpc_model.pi()` and adds a behavior-cloning MSE against the selected dataset
action. If `joint_training.tdmpc_consistency_weight > 0` and `za_latent_index_mode`
returns a sequence, the trainer additionally applies a TD-MPC dynamics
consistency MSE with `tdmpc_model.next(z_t, action_t)` against `z_{t+1}`.
The full `tdmpc2.update()` reward/value loss is not called here because the
converted LeRobot-like dataset does not currently provide the replay-buffer
`reward/done` fields that `tdmpc2.py` expects.

To return to the old adaptor-only training, set:

```yaml
train_mode: 'za_adaptor_only'
```

The script runs:

```bash
torchrun main.py \
  --config_file configs/ltx_model/za_adaptor_tdmpc.yaml \
  --runner_class_path runner/za_adaptor_trainer.py \
  --runner_class ZaAdaptorTrainer
```

Checkpoints are saved as:

```text
OUTPUTS/ltx/za_adaptor/<time_or_subfolder>/step_<global_step>/za_adaptor.pt
```

## Important Configs

```yaml
za_adaptor:
  input_dim: 2048
  output_dim: 512
  size: base
  pool: mean
  target_pool: last
  timestep_mode: clean

beta_fusion:
  enabled: true
  beta_dim: 5
  fusion_dim: 512
  num_layers: 1
```

Use `timestep_mode: clean` when fitting Za from real data latents. Use
`timestep_mode: random` only if you intentionally want the adaptor to see the
same noisy latent distribution used during diffusion training.

If Za changes from 512 dimensions, update:

```yaml
za_adaptor:
  output_dim: <new_dim>
```
