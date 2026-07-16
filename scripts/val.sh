export CUDA_VISIBLE_DEVICES=2
python scripts/validate_action_conditioning.py \
    --config_file configs/ltx_model/video_model_lerobot.yaml \
    --checkpoint_path OUTPUTS/ltx/2026_07_15_17_47_22/step_60000 \
    --output_path OUTPUTS/action_val_ep0 \
    --episode_id 0 \
    --scale_factor 2.5 \
    --mirror_axis 0 \
    --start_frame 75