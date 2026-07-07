CUDA_VISIBLE_DEVICES=1 bash scripts/infer.sh \
    main.py \
    configs/ltx_model/policy_model_lerobot.yaml \
    /mnt/pfs/s7fsio/code/pyl/tau-0/Genie-Envisioner/OUTPUTS/ltx/action_video/2026_07_06_17_14_45/step_40000/diffusion_pytorch_model.safetensors \
    test_result/ltx_action \
    move_to_object \
    29512 \
    50