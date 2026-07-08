CUDA_VISIBLE_DEVICES=1 bash scripts/infer.sh \
    main.py \
    configs/ltx_model/policy_model_lerobot.yaml \
    OUTPUTS/ltx/action_video/2026_07_07_16_31_47/step_20000/diffusion_pytorch_model.safetensors \
    test_result/ltx_action_video \
    habitat_lerobot \
    29512 \
    100