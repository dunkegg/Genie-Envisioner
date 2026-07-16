import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd
import torch
import cv2
from einops import rearrange
from yaml import load, Loader
from typing import List

from utils import import_custom_class, save_video
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model
from data.lerobot_like_dataset import overlay_trajectory_on_video_tensor

def load_config(config_file):
    cd = load(open(config_file, "r"), Loader=Loader)
    args = argparse.Namespace(**cd)
    return args


def prepare_model(args, dtype=torch.bfloat16, device="cuda:0"):
    tokenizer_class = import_custom_class(
        args.tokenizer_class, getattr(args, "tokenizer_class_path", "transformers")
    )
    textenc_class = import_custom_class(
        args.textenc_class, getattr(args, "textenc_class_path", "transformers")
    )
    cond_models = load_condition_models(
        tokenizer_class, textenc_class,
        args.pretrained_model_name_or_path,
        load_weights=args.load_weights
    )
    tokenizer = cond_models["tokenizer"]
    text_encoder = cond_models["text_encoder"].to(device, dtype=dtype).eval()

    vae_class = import_custom_class(
        args.vae_class, getattr(args, "vae_class_path", "transformers")
    )
    if getattr(args, 'vae_path', False):
        vae = load_vae_models(vae_class, args.vae_path).to(device, dtype=dtype).eval()
    else:
        vae = load_latent_models(vae_class, args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
    if isinstance(vae.latents_mean, List):
        vae.latents_mean = torch.FloatTensor(vae.latents_mean)
    if isinstance(vae.latents_std, List):
        vae.latents_std = torch.FloatTensor(vae.latents_std)
    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    diffusion_model_class = import_custom_class(
        args.diffusion_model_class, getattr(args, "diffusion_model_class_path", "transformers")
    )
    diffusion_model = load_diffusion_model(
        model_cls=diffusion_model_class,
        model_dir=args.diffusion_model['model_path'],
        load_weights=args.load_weights,
        **args.diffusion_model['config']
    ).to(device, dtype=dtype)

    #覆盖yaml中的config到默认config
    for key, value in args.diffusion_model['config'].items():
        diffusion_model.config[key] = value

    diffusion_scheduler_class = import_custom_class(
        args.diffusion_scheduler_class, getattr(args, "diffusion_scheduler_class_path", "diffusers")
    )
    if hasattr(args, "diffusion_scheduler_args"):
        scheduler = diffusion_scheduler_class(**args.diffusion_scheduler_args)
    else:
        scheduler = diffusion_scheduler_class()

    pipeline_class = import_custom_class(
        args.pipeline_class, getattr(args, "pipeline_class_path", "diffusers")
    )
    pipe = pipeline_class(
        scheduler=scheduler, vae=vae,
        text_encoder=text_encoder, tokenizer=tokenizer,
        transformer=diffusion_model
    )
    return pipe, vae

def load_episode_data(args, episode_id, start_frame, device, dtype):
    """从parquet和video加载一个episode的数据"""
    data_root = args.data['val']['data_roots'][0]
    domain = args.data['val']['domains'][0]
    valid_cams = args.data['val']['valid_cam']
    n_prev = args.data['train']['n_previous']
    sample_size = args.data['train']['sample_size']  # [H, W]
    h, w = sample_size[0], sample_size[1]

    # 读parquet
    import json as jsonlib
    meta_path = os.path.join(data_root, domain, 'meta', 'info.json')
    with open(meta_path) as f:
        info = jsonlib.load(f)
    chunks_size = info['chunks_size']
    episode_chunk = episode_id // chunks_size
    parquet_path = os.path.join(
        data_root, domain, 'data',
        f'chunk-{episode_chunk:03d}',
        f'episode_{episode_id:06d}.parquet'
    )
    df = pd.read_parquet(parquet_path)
    actions = np.stack([df[args.data['val']['action_key']][i] for i in range(start_frame, len(df))]).astype(np.float32)
    raw_states = np.stack([df[args.data['val']['state_key']][i] for i in range(start_frame, len(df))]).astype(np.float32)

    # 读取caption
    tasks_path = os.path.join(data_root, domain, 'meta', 'tasks.jsonl')
    episodes_path = os.path.join(data_root, domain, 'meta', 'episodes.jsonl')
    with open(episodes_path) as f:
        episodes = [json.loads(l) for l in f if l.strip()]
    ep_info = episodes[episode_id]
    task_index = ep_info['tasks'][0] if isinstance(ep_info['tasks'][0], int) else 0
    with open(tasks_path) as f:
        tasks = {json.loads(l)['task_index']: json.loads(l)['task'] for l in f if l.strip()}
    caption = tasks.get(task_index, "robot manipulation task")

    # 读取视频帧（只取前n_prev帧作为memory）
    from moviepy.editor import VideoFileClip
    mv_images = []
    for cam in valid_cams:
        video_path = os.path.join(
            data_root, domain, 'videos',
            f'chunk-{episode_chunk:03d}',
            cam,
            f'episode_{episode_id:06d}.mp4'
        )
        reader = VideoFileClip(video_path)
        fps = reader.fps
        frames = []
        for i in range(n_prev):
            frame_time = float(start_frame + i) / fps
            frame = reader.get_frame(frame_time)
            
            frame = cv2.resize(frame, (w, h))
            frame = frame.astype(np.float32) / 255.0 * 2.0 - 1.0
            frame = torch.from_numpy(np.transpose(frame, (2, 0, 1)))
            frames.append(frame)
        reader.close()
        mv_images.append(torch.stack(frames, dim=1))  # c,t,h,w

    # v,c,t,h,w -> (v) c t h w
    mv_images = torch.stack(mv_images, dim=0).to(device, dtype=dtype)

    return mv_images, actions, raw_states, caption


def transform_trajectory(trajectory, history_len, mode, scale_factor=2.5, mirror_axis=0):
    new_traj = trajectory.copy()
    if len(trajectory) <= history_len:
        return new_traj
    anchor = trajectory[history_len - 1]
    future_traj = trajectory[history_len:]
    deltas = future_traj - anchor
    if mode == 'scale':
        new_deltas = deltas * scale_factor
    elif mode == 'mirror':
        new_deltas = deltas.copy()
        new_deltas[:, mirror_axis] = -new_deltas[:, mirror_axis]
    else:
        new_deltas = deltas
    new_traj[history_len:] = anchor + new_deltas
    return new_traj


def run_inference(pipe, image, actions, caption, args, device, dtype, tag, output_path):
    """跑一次推理并保存视频"""
    n_prev = args.data['train']['n_previous']
    chunk = args.data['train']['chunk']
    action_chunk = args.data['train']['action_chunk']
    h, w = args.data['train']['sample_size']
    n_view = len(args.data['val']['valid_cam'])
    TEMPORAL_DOWN = pipe.vae_temporal_compression_ratio

    # 计算motion delta
    if getattr(args, 'use_motion_conditioning', False):
        actions_t = torch.FloatTensor(actions).unsqueeze(0).to(device, dtype=dtype)
        motion_deltas = actions_t[:, 1:] - actions_t[:, :-1]  # [1, T-1, action_dim]
    else:
        motion_deltas = None

    if motion_deltas is not None:
        print(f"motion_deltas shape: {motion_deltas.shape}")
        print(f"motion_deltas range: [{motion_deltas.min():.4f}, {motion_deltas.max():.4f}]")
    # if motion_deltas is not None:
    #     print(f"  motion_deltas shape: {motion_deltas.shape}, range: [{motion_deltas.min():.4f}, {motion_deltas.max():.4f}]")
    # else:
    #     print("  motion_deltas is None!")

    # image: [v, c, t, h, w] -> [(v), c, t, h, w]已经是正确格式
    prompt = [caption] if getattr(args, 'use_text_condition', True) else ['']
    print(f"  Running inference: {tag}")
    with torch.no_grad():
        preds = pipe.infer(
            image=image,
            prompt=prompt,
            negative_prompt='',
            num_inference_steps=args.num_inference_step,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=7.0,
            height=h, width=w,
            n_view=n_view,
            n_prev=n_prev,
            chunk=(chunk - 1) // TEMPORAL_DOWN + 1,
            return_video=True,
            return_action=False,
            noise_seed=None,
            action_chunk=action_chunk,
            pixel_wise_timestep=args.pixel_wise_timestep,
            n_chunk=1,
            motion_deltas=motion_deltas,
        )[0]

    video = preds['video'].data.cpu()
    fps = int(getattr(args, 'basic_fps', 30) / (action_chunk // chunk))
    save_path = os.path.join(output_path, f'video_{tag}.mp4')
    save_video(
        rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)[0],
        save_path, fps=fps
    )
    print(f"  Saved: {save_path}")
    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--episode_id', type=int, default=0)
    parser.add_argument('--scale_factor', type=float, default=2.5)
    parser.add_argument('--mirror_axis', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--start_frame', type=int, default=0)
    cli_args = parser.parse_args()

    import json
    args = load_config(cli_args.config_file)
    # 覆盖checkpoint路径
    args.diffusion_model['model_path'] = cli_args.checkpoint_path

    os.makedirs(cli_args.output_path, exist_ok=True)
    device = cli_args.device
    dtype = torch.bfloat16

    print("Loading models...")
    pipe, vae = prepare_model(args, dtype=dtype, device=device)

    print(f"Loading episode {cli_args.episode_id}...")
    if args.data['train']['use_trajectory_condition']:
        image, actions, raw_states, caption = load_episode_data(args, cli_args.episode_id, cli_args.start_frame, device, dtype)
        n_prev = args.data['train']['n_previous']
        traj_cfg = args.data['train']
        last_mem_idx = n_prev - 1 
        states_orig     = raw_states.copy()
        states_scaled   = transform_trajectory(raw_states, n_prev, 'scale',  cli_args.scale_factor)
        states_mirrored = transform_trajectory(raw_states, n_prev, 'mirror', mirror_axis=cli_args.mirror_axis)
        image_cvthw = image.permute(1, 0, 2, 3, 4).cpu()  # (C, V, T, H, W)
        
        from data.lerobot_like_dataset import overlay_trajectory_on_video_tensor

        def make_cond_frame(states):
            mem_with_traj = overlay_trajectory_on_video_tensor(
                mem_video_tensor=image_cvthw,
                raw_states=states,
                current_frame_idx=last_mem_idx,
                draw_steps=traj_cfg.get('trajectory_steps', 45),
                hfov=traj_cfg.get('camera_hfov', 87.0),
                sensor_height=traj_cfg.get('camera_sensor_height', 0.5),
                alpha=traj_cfg.get('trajectory_alpha', 0.75),
            )
            # 转回 (V, C, T, H, W) 给 pipeline
            return mem_with_traj.permute(1, 0, 2, 3, 4).to(device, dtype=dtype)

        image_orig     = make_cond_frame(states_orig)
        image_scaled   = make_cond_frame(states_scaled)
        image_mirrored = make_cond_frame(states_mirrored)

        # 保存条件帧对比图
        for tag, img in [('orig', image_orig), ('scaled', image_scaled), ('mirrored', image_mirrored)]:
            frame = img[0, :, -1].float().cpu()  # 第0个view，最后一个mem帧
            frame_np = ((frame.numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cli_args.output_path, f'cond_{tag}.jpg'), frame_bgr)
        print("Saved condition frame comparison images")

        # 推理
        video_orig     = run_inference(pipe, image_orig,     None, caption, args, device, dtype, 'original', cli_args.output_path)
        video_scaled   = run_inference(pipe, image_scaled,   None, caption, args, device, dtype, f'scaled_x{cli_args.scale_factor}', cli_args.output_path)
        video_mirrored = run_inference(pipe, image_mirrored, None, caption, args, device, dtype, f'mirrored_axis{cli_args.mirror_axis}', cli_args.output_path)
        
        image_black = torch.zeros_like(image_orig)  # 全黑条件帧
        video_black = run_inference(pipe, image_black, None, caption, args, device, dtype, 'black', cli_args.output_path)

        diff_black = (video_orig - video_black).abs().mean().item()
        print(f"  original vs black frame: mean abs diff = {diff_black:.4f}")

    else:
        image, actions, raw_states, caption = load_episode_data(args, cli_args.episode_id, cli_args.start_frame, device, dtype)
        n_prev = args.data['train']['n_previous']
        print(f"  Caption: {caption}")
        print(f"  Actions shape: {actions.shape}, range: [{actions.min():.3f}, {actions.max():.3f}]")

        # 保存action对比图
        import matplotlib.pyplot as plt
        actions_scaled = transform_trajectory(actions, n_prev, 'scale', cli_args.scale_factor)
        actions_mirrored = transform_trajectory(actions, n_prev, 'mirror', mirror_axis=cli_args.mirror_axis)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        T = len(actions)
        for ax, act, title in zip(axes,
            [actions, actions_scaled, actions_mirrored],
            ['Original', f'Scaled x{cli_args.scale_factor}', f'Mirrored axis={cli_args.mirror_axis}']
        ):
            ax.plot(act[:, 0], label='dim0')
            ax.plot(act[:, 1], label='dim1')
            ax.axvline(x=n_prev, color='r', linestyle='--', label='anchor')
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(cli_args.output_path, 'action_comparison.png'), dpi=150)
        plt.close()
        print("Saved action comparison plot")

        # 跑三组推理
        video_orig = run_inference(pipe, image, actions, caption, args, device, dtype,
                                'original', cli_args.output_path)
        video_scaled = run_inference(pipe, image, actions_scaled, caption, args, device, dtype,
                                    f'scaled_x{cli_args.scale_factor}', cli_args.output_path)
        video_mirrored = run_inference(pipe, image, actions_mirrored, caption, args, device, dtype,
                                    f'mirrored_axis{cli_args.mirror_axis}', cli_args.output_path)

    # 计算差异
    diff_scaled = (video_orig - video_scaled).abs().mean().item()
    diff_mirrored = (video_orig - video_mirrored).abs().mean().item()
    print(f"\nResults:")
    print(f"  original vs scaled:   mean abs diff = {diff_scaled:.4f}")
    print(f"  original vs mirrored: mean abs diff = {diff_mirrored:.4f}")

    if diff_scaled < 0.01 and diff_mirrored < 0.01:
        print("  WARNING: differences very small, motion conditioning may not be effective")
    else:
        print("  motion conditioning appears to be working")

    # 保存结果摘要
    summary = {
        'episode_id': cli_args.episode_id,
        'caption': caption,
        'scale_factor': cli_args.scale_factor,
        'mirror_axis': cli_args.mirror_axis,
        'diff_original_vs_scaled': diff_scaled,
        'diff_original_vs_mirrored': diff_mirrored,
    }
    with open(os.path.join(cli_args.output_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {cli_args.output_path}/summary.json")

    print(f"Loading checkpoint from: {args.diffusion_model['model_path']}")
if __name__ == '__main__':
    main()