# WorldVLN video backbone for Za / BEV training

The Za trainer can replace its original LTX diffusion backbone with the video predictor from the separate WorldVLN project.

## Exact model boundary

Loaded from WorldVLN:

- InfinityStar `Infinity` autoregressive video transformer
- InfinityStar VideoVAE
- official dynamic-resolution schedule and visual transformer feature path

Not loaded:

- WorldVLN TSformer
- latent-to-action decoder
- GRPO server or action head

The data path is:

```text
history RGB
  -> WorldVLN VideoVAE visual tokens
  -> one frozen InfinityStar 36-layer visual transformer pass
  -> final intermediate hidden states [B,L,4096]
  -> token-axis adaptive pooling [B,8,4096]
  -> trainable 4096-to-2048 projector = Zb tokens
  -> BetaZBFusion(beta cross-attn)
  -> ZaAdaptor = Za-prime
  -> BEV DiT / TD-MPC losses
```

InfinityStar and its VAE are frozen. In `feature_mode: intermediate`, T5 is not loaded and the 159-stage autoregressive future-video generation is skipped. These features describe the observed video and are not predicted-future latents. The feature projector, BetaZBFusion, ZaAdaptor, and BEV DiT are trainable.

Set `feature_mode: predicted_latent` and `latent_channels: 64` only when the old full autoregressive prediction path is explicitly required.

## Configuration

`configs/ltx_model/za_adaptor_tdmpc.yaml` now points to the local WorldVLN assets and uses a separate output root:

```text
OUTPUTS/worldvln/za_adaptor_tdmpc
```

This prevents `resume_from_checkpoint: latest` from loading an incompatible LTX-backbone checkpoint. When resuming an older WorldVLN predicted-latent checkpoint, Za/Beta/BEV/TD-MPC states are restored but the incompatible 64-channel projector is skipped and the new 4096-channel projector starts from scratch.

The released InfinityStar inference implementation is cache-oriented and uses batch size one. The config therefore uses `batch_size: 1`. Each distributed rank loads one frozen 8B model replica.

## Environment

Run from a Python 3.10 environment containing dependencies from both projects. In particular, WorldVLN requires `timm`; the currently inspected conda environments do not provide it. Install the WorldVLN requirements in the intended training environment before launching:

```bash
cd /home/pengyulin/h20_2_code/pyl/worldWAM/WorldVLN.code
pip install -r requirements.txt
```

The project also expects a compatible PyTorch/CUDA build (WorldVLN recommends PyTorch 2.5.1).

WorldVLN paths are resolved relative to the Genie-Envisioner checkout, so both `/home/...` and `/mnt/pfs/...` mounts work. If the repositories are not siblings under `pyl/`, override the root explicitly:

```bash
export WORLDVLN_ROOT=/absolute/path/to/WorldVLN.code
```

## Launch

Use the existing Za trainer launcher:

```bash
cd /home/pengyulin/h20_2_code/pyl/tau-0/Genie-Envisioner
NPROC_PER_NODE=1 bash scripts/train_za_adaptor.sh \
  main.py \
  configs/ltx_model/za_adaptor_tdmpc.yaml
```

Start with one rank because each rank loads the full InfinityStar checkpoint. Increase `NPROC_PER_NODE` only after confirming per-GPU memory usage.

## Expected startup evidence

The log should include:

```text
Replacing the LTX video backbone with frozen WorldVLN InfinityStar
[you selected Infinity with infinity_qwen8b] model size: ...B
WorldVLN latent channel mismatch ...   # must NOT appear
WorldVLN feature mode=intermediate, feature_dim=4096, output_tokens=8
```

The first real batch validates the intermediate hidden-state contract. There should no longer be a `159/159` autoregressive sampling progress bar during each training step.
