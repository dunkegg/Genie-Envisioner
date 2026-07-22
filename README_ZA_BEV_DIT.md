# Za-prime BEV DiT

This branch predicts the future BEV sequence after Za-prime:

`beta -> BetaZBFusion (cross-attn) -> ZaAdaptor -> Za-prime -> ZaBEVDiT -> BEV flow velocity`

The frozen video backbone produces Zb. The BEV loss is intentionally not detached at Za-prime, so it trains `ZaBEVDiT`, `ZaAdaptor`, and all of `BetaZBFusion`, including `BetaZBFusionBlock.cross_attn`. The video backbone remains frozen.

## 1. Extract BEV from H5

Raw episodes contain `traj_*/bev` as `[T,128,256] uint8`, aligned one-to-one with `traj_*/rgb`. Convert the dataset again:

```bash
cd /home/pengyulin/h20_2_code/pyl/tau-0/Genie-Envisioner
python h5_tdmpc_2_lerobot.py \
  --input_dir isaac_z_dataset_7_10 \
  --output_dir lerobot_data/tdmpc_za_7_10 \
  --domain_name tdmpc_za \
  --bev_key bev \
  --bev_column bev_map
```

Each BEV frame is saved losslessly as an `.npy` file under `bev/chunk-*/episode_*`. Its relative path is stored in the parquet `bev_map` column. Existing RGB, latent, beta, state, and action fields are preserved.

## 2. Frame alignment

The dataset loads future BEV with the exact same `vid_indexes` as future RGB. With the current config:

- `action_chunk: 108`
- `chunk: 18`
- temporal stride: `108 / 18 = 6`

Both RGB and BEV supervision therefore use the same 18 future timestamps. BEV values are reversibly normalized from raw uint8 to `[-1,1]`; `ZaBEVDiT.to_dataset_uint8()` converts reconstructed maps back to the original dataset format.

## 3. Train

The settings are in `configs/ltx_model/za_adaptor_tdmpc.yaml`. Start training with the existing launcher:

```bash
bash scripts/train_za_adaptor.sh main.py configs/ltx_model/za_adaptor_tdmpc.yaml
```

BEV keeps the same flow-matching definition and adds direct pixel/edge reconstruction supervision:

```text
x_t = (1 - sigma) * bev + sigma * noise
target_v = noise - bev
pred_x0 = x_t - sigma * pred_v
loss = flow_mse + 2.0 * pixel_mse + 0.5 * pixel_l1 + edge_l1
```

The high-fidelity head uses 8x8 DiT patches, 12 transformer blocks, and a four-block full-resolution residual CNN. TensorBoard reports `bev_flow_mse`, `bev_pixel_mse`, `bev_pixel_l1`, and `bev_edge_l1` separately. A lower flow loss alone is not treated as proof of pixel-level reconstruction quality.

Checkpoints remain at `step_*/za_adaptor.pt` and now also contain `za_bev_dit_state_dict` and its config.

## 4. Validation inference

Every `steps_to_val` training steps, rank 0 runs iterative BEV denoising on the validation set. The number of denoising steps and samples is controlled by:

```yaml
za_bev_dit:
  validation_inference_steps: 50
  validation_samples: 1
```

Results are saved under:

```text
OUTPUTS/ltx/za_adaptor_tdmpc/<run>/Validation_step_<step>/bev/
```

For every future frame it writes `*_pred.png`, `*_gt.png`, and `*_gt_pred.png` (ground truth on the left, prediction on the right). It also writes `*_sequence_gt_top_pred_bottom.png`, with all 18 ground-truth frames on the top row and all predictions on the bottom row.

## 5. Resume from the latest checkpoint

The training config enables automatic resume:

```yaml
resume_from_checkpoint: latest
```

At startup the trainer scans `output_dir/*/step_*/za_adaptor.pt` and chooses the most recently saved checkpoint. New checkpoints include model weights, optimizer state, LR scheduler state, and `global_step`; the progress bar and validation/save schedules continue from that step. When loading an old 16x16/8-layer BEV checkpoint, compatible transformer weights are retained while the new 8x8 patch layers, four extra blocks, and pixel refinement head are initialized from scratch.

To disable resume, set `resume_from_checkpoint: null`. To resume a specific checkpoint, set it to either a `step_*` directory or its `za_adaptor.pt` path.
