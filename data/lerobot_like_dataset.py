import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import traceback
import json
import random
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
import glob
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from PIL import Image

from data.utils.statistics import StatisticInfo
from utils import zero_rank_print
from data.utils.utils import intrinsic_transform, gen_crop_config, intrin_crop_transform


def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='UTF-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ============================================================
# 轨迹投影核心函数（基于用户验证的逻辑）
# ============================================================

def compute_camera_axes(states, current_frame_idx):
    """
    用运动矢量推导相机朝向（前向/右向单位向量）
    states: np.ndarray (T, state_dim)，前两列是全局 X, Z
    返回: (f_x, f_z, r_x, r_z) 前向和右向单位向量
    """
    base_x = states[current_frame_idx][0]
    base_z = states[current_frame_idx][1]

    target_next = min(current_frame_idx + 1, len(states) - 1)
    next_x = states[target_next][0]
    next_z = states[target_next][1]

    fw_x = next_x - base_x
    fw_z = next_z - base_z
    length = np.sqrt(fw_x ** 2 + fw_z ** 2)

    if length > 1e-5:
        f_x = fw_x / length
        f_z = fw_z / length
    else:
        f_x, f_z = 0.0, 1.0  # 默认朝正前方

    # 右向向量（顺时针旋转 90°）
    r_x = f_z
    r_z = -f_x

    return f_x, f_z, r_x, r_z


def project_future_trajectory(
    states,
    current_frame_idx,
    draw_steps,
    image_w,
    image_h,
    hfov=87.0,
    sensor_height=0.5,
):
    """
    把未来 draw_steps 步的 state 轨迹投影到图像坐标。
    states: np.ndarray (T, state_dim)，前两列是全局 X, Z（原始未归一化）
    返回: list of (u, v) 像素坐标
    """
    fx = (image_w / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    fy = fx
    cx, cy = image_w / 2.0, image_h / 2.0

    base_x = states[current_frame_idx][0]
    base_z = states[current_frame_idx][1]

    f_x, f_z, r_x, r_z = compute_camera_axes(states, current_frame_idx)

    projected = []
    target_len = min(len(states), current_frame_idx + draw_steps + 1)

    for t in range(current_frame_idx + 1, target_len):
        world_dx = states[t][0] - base_x
        world_dz = states[t][1] - base_z

        # 全局位移 -> 相机局部坐标
        X_cam = world_dx * r_x + world_dz * r_z
        Z_cam = world_dx * f_x + world_dz * f_z
        Y_cam = sensor_height

        if Z_cam <= 0.05:
            continue

        u = int((fx * X_cam / Z_cam) + cx)
        v = int((fy * Y_cam / Z_cam) + cy)
        # 允许轻微超出屏幕（连线时裁剪），但过远的点丢弃
        if -image_w < u < 2 * image_w and -image_h < v < 2 * image_h:
            projected.append((u, v))

    return projected


def draw_trajectory_on_frame_bgr(canvas_bgr, projected_points, alpha=0.75):
    """
    在 BGR uint8 图像上绘制轨迹（红近绿远），半透明叠加。
    """
    if len(projected_points) < 2:
        return canvas_bgr

    overlay = canvas_bgr.copy()
    n = len(projected_points)

    for i in range(n - 1):
        pt1 = projected_points[i]
        pt2 = projected_points[i + 1]
        ratio = i / max(n - 1, 1)
        color = (0, int(255 * ratio), int(255 * (1 - ratio)))  # BGR: 红->绿
        cv2.line(overlay, pt1, pt2, color, thickness=3, lineType=cv2.LINE_AA)
        cv2.circle(overlay, pt1, radius=4, color=color, thickness=-1)

    # 终点画大圆
    cv2.circle(overlay, projected_points[-1], radius=6, color=(0, 255, 0), thickness=-1)

    return cv2.addWeighted(overlay, alpha, canvas_bgr, 1 - alpha, 0)


def overlay_trajectory_on_video_tensor(
    mem_video_tensor,   # (C, V, T_mem, H, W), float, -1~1
    raw_states,         # np.ndarray (T_total, state_dim), 未归一化原始值
    current_frame_idx,  # 当前帧在全局 states 里的索引（用于计算相机朝向）
    draw_steps=45,
    hfov=87.0,
    sensor_height=0.5,
    alpha=0.75,
):
    """
    在 mem 帧上叠加轨迹，返回同 shape 的 tensor（-1~1）。
    """
    C, V, T_mem, H, W = mem_video_tensor.shape

    projected = project_future_trajectory(
        states=raw_states,
        current_frame_idx=current_frame_idx,
        draw_steps=draw_steps,
        image_w=W,
        image_h=H,
        hfov=hfov,
        sensor_height=sensor_height,
    )

    if len(projected) < 2:
        return mem_video_tensor  # 没有有效投影点，直接返回原始

    result = mem_video_tensor.clone()

    for iv in range(V):
        # 只处理最后一个 mem 帧，其余帧保持原样
        it = T_mem - 1
        frame_chw = mem_video_tensor[:, iv, it]
        frame_np = ((frame_chw.float().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        frame_bgr_with_traj = draw_trajectory_on_frame_bgr(frame_bgr, projected, alpha=alpha)
        frame_rgb_out = cv2.cvtColor(frame_bgr_with_traj, cv2.COLOR_BGR2RGB)
        result[:, iv, it] = torch.from_numpy(frame_rgb_out.transpose(2, 0, 1)).float() / 127.5 - 1.0

    return result


# ============================================================
# 调试保存工具
# ============================================================

def save_trajectory_debug_image(
    frame_tensor,        # (C, H, W), float, -1~1
    raw_states,
    current_frame_idx,
    draw_steps,
    save_path,
    hfov=87.0,
    sensor_height=0.5,
):
    """保存单帧轨迹可视化图片，用于肉眼验证效果。"""
    C, H, W = frame_tensor.shape
    frame_np = ((frame_tensor.float().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

    projected = project_future_trajectory(
        raw_states, current_frame_idx, draw_steps, W, H, hfov, sensor_height
    )
    frame_bgr_out = draw_trajectory_on_frame_bgr(frame_bgr, projected)
    cv2.imwrite(save_path, frame_bgr_out)
    print(f"[DEBUG] 轨迹图保存至: {save_path}  (投影点数: {len(projected)})")


# def save_trajectory_debug_video(
#     video_tensor,        # (C, V, T, H, W), float, -1~1，已叠加轨迹
#     save_path,
#     fps=10,
#     view_idx=0,
# ):
#     """把带轨迹的 mem 帧序列保存为小视频，方便查看效果。"""
#     C, V, T, H, W = video_tensor.shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
#     for t in range(T):
#         frame = video_tensor[:, view_idx, t]  # (C, H, W)
#         frame_np = ((frame.float().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
#         frame_bgr = cv2.cvtColor(frame_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
#         writer.write(frame_bgr)
#     writer.release()
#     print(f"[DEBUG] 轨迹视频保存至: {save_path}")
def save_trajectory_debug_video(
    video_tensor,
    save_path,
    fps=10,
    view_idx=0,
):
    C, V, T, H, W = video_tensor.shape
    tmp_path = save_path.replace(".mp4", "_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))
    for t in range(T):
        frame = video_tensor[:, view_idx, t]
        frame_np = ((frame.float().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()

    # 尝试用 ffmpeg 转 H.264，失败就直接用 mp4v
    ret = os.system(f"ffmpeg -y -i {tmp_path} -vcodec libx264 -pix_fmt yuv420p {save_path} -loglevel quiet")
    if os.path.exists(tmp_path):
        if ret == 0:
            os.remove(tmp_path)
        else:
            # ffmpeg 失败，直接把 tmp 当结果
            os.rename(tmp_path, save_path)
    
    print(f"[DEBUG] 轨迹视频保存至: {save_path}")

# ============================================================
# Dataset
# ============================================================

class CustomLeRobotDataset(Dataset):
    def __init__(self,
        data_roots,
        domains,
        task_recap_file=None,
        step_recap_file=None,
        sample_size=(192, 256),
        sample_n_frames=64,
        preprocess='resize',
        valid_cam=['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right'],
        chunk=1,
        action_chunk=None,
        n_previous=-1,
        previous_pick_mode='uniform',
        random_crop=True,
        dataset_info_cache_path=None,
        action_type="absolute",
        action_space="joint",
        ignore_seek=False,
        train_dataset=True,
        action_key="action",
        state_key="observation.state",
        use_unified_prompt=False,
        unified_prompt="best quality, consistent and smooth motion, realistic, clear and distinct.",
        fix_epiidx=None,
        fix_sidx=None,
        fix_mem_idx=None,
        stat_file=None,
        # ---- 轨迹条件新增参数 ----
        use_trajectory_condition=False,
        trajectory_steps=45,         # 只显示未来多少步的轨迹
        camera_hfov=87.0,
        camera_sensor_height=0.5,
        trajectory_alpha=0.75,       # 轨迹叠加透明度
        trajectory_debug_dir=None,   # 不为 None 时保存调试图/视频
        trajectory_debug_interval=200,  # 每隔多少个样本保存一次调试图
    ):
        zero_rank_print(f"loading annotations...")

        assert(action_type in ["delta", "absolute", "relative"])
        self.action_type = action_type
        assert(action_space in ["eef", "joint"])
        self.action_space = action_space

        self.action_key = action_key
        self.state_key = state_key
        self.random_crop = random_crop

        # ---- 轨迹参数 ----
        self.use_trajectory_condition = use_trajectory_condition
        self.trajectory_steps = trajectory_steps
        self.camera_hfov = camera_hfov
        self.camera_sensor_height = camera_sensor_height
        self.trajectory_alpha = trajectory_alpha
        self.trajectory_debug_dir = trajectory_debug_dir
        self.trajectory_debug_interval = trajectory_debug_interval
        self._debug_counter = 0  # 内部计数，控制调试保存频率

        if trajectory_debug_dir is not None:
            os.makedirs(trajectory_debug_dir, exist_ok=True)

        if not isinstance(valid_cam, (list, tuple)):
            valid_cam = [valid_cam, ]
        self.valid_cam = valid_cam
        if len(data_roots) == 1 and len(domains) > 1:
            data_roots = data_roots * len(domains)
        self.data_roots = data_roots
        self.dataset = []

        if dataset_info_cache_path is not None and os.path.exists(dataset_info_cache_path):
            zero_rank_print(f"Load Cache Dataset Information from {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "r") as f:
                self.dataset = json.load(f)
        else:
            for _data_root, _domain_name in zip(self.data_roots, domains):
                print(f"Loading {_domain_name} data from {_data_root}")
                meta_folder = os.path.join(_data_root, _domain_name, "meta")
                data_folder = os.path.join(_data_root, _domain_name, "data")
                video_folder = os.path.join(_data_root, _domain_name, "videos")

                tasks_jsonl = os.path.join(meta_folder, "tasks.jsonl")
                task_index_task_str = load_jsonl(tasks_jsonl)
                task_index_task_str_dict = {}
                for item in task_index_task_str:
                    task_index_task_str_dict[item['task_index']] = item['task']

                with open(os.path.join(meta_folder, "info.json"), "r") as f:
                    metainfo = json.load(f)
                    total_chunks = metainfo["total_chunks"]
                    chunks_size = metainfo["chunks_size"]

                episodes_jsonl = os.path.join(meta_folder, "episodes.jsonl")
                epiosdes_data = load_jsonl(episodes_jsonl)

                for episode_data in tqdm(epiosdes_data):
                    episode_index = episode_data['episode_index']
                    tasks = episode_data['tasks']
                    if len(tasks) > 1:
                        task = random.choice(tasks)
                    else:
                        task = tasks[0]
                    length = episode_data['length']
                    episode_chunk = int(episode_index // chunks_size)
                    parquet_path = os.path.join(data_folder, f"chunk-{episode_chunk:03d}", f"episode_{episode_index:06d}.parquet")
                    if not os.path.exists(parquet_path):
                        zero_rank_print(f"parquet file not found: {parquet_path}")
                        continue
                    video_path = os.path.join(video_folder, f"chunk-{episode_chunk:03d}", "{}", f"episode_{episode_index:06d}.mp4")
                    info = [
                        video_path, None, parquet_path,
                        _domain_name, "", None, task, length,
                    ]
                    self.dataset.append(info)

        if dataset_info_cache_path is not None and not(os.path.exists(dataset_info_cache_path)):
            zero_rank_print(f"Save Cache Dataset Information to {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "w") as f:
                json.dump(self.dataset, f)

        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.chunk = chunk
        if action_chunk is None:
            action_chunk = chunk
        self.action_chunk = action_chunk
        self.video_temporal_stride = self.action_chunk // self.chunk
        assert(self.chunk * self.video_temporal_stride == self.action_chunk)

        self.sample_n_frames = sample_n_frames
        self.sample_size = sample_size

        if preprocess == 'center_crop_resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),
                transforms.CenterCrop(sample_size),
            ])
        if preprocess == 'resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.preprocess = preprocess

        if n_previous > 1:
            self.n_previous = int(n_previous)
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = int(self.sample_n_frames - self.chunk)
            self.previous_pick_mode = 'uniform'

        if task_recap_file is not None:
            with open(task_recap_file, 'r', encoding='UTF-8') as f:
                self.task_recap_map = json.load(f)
        else:
            self.task_recap_map = None

        if step_recap_file is not None:
            with open(step_recap_file, 'r', encoding='UTF-8') as f:
                self.step_recap_map = json.load(f)
        else:
            self.step_recap_map = None

        self.use_unified_prompt = use_unified_prompt
        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = fix_mem_idx

        self.StatisticInfo = StatisticInfo
        if stat_file is not None:
            with open(stat_file, "r") as f:
                self.StatisticInfo = json.load(f)

        self.ignore_seek = ignore_seek


    def get_frame_indexes(self, total_frames):
        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx + self.action_chunk))
            frame_indexes = action_indexes[::self.video_temporal_stride]
            action_indexes = np.clip(action_indexes, a_min=0, a_max=total_frames - 1)
            frame_indexes = np.clip(frame_indexes, a_min=0, a_max=total_frames - 1)
            fix_mem_idx = np.asarray(self.fix_mem_idx).reshape(-1)
            fix_mem_idx = np.clip(fix_mem_idx, a_min=0, a_max=total_frames - 1).astype(np.int64)
            start_idx = int(fix_mem_idx[-1])
            vid_indexes = np.concatenate([fix_mem_idx, start_idx + frame_indexes])
            act_indexes = start_idx + action_indexes
            return vid_indexes, act_indexes

        chunk_end = random.randint(self.action_chunk, total_frames + self.action_chunk)
        indexes_start = max(-self.n_previous, chunk_end - self.sample_n_frames)
        indexes = np.array(list(range(indexes_start, chunk_end)))
        indexes = np.clip(indexes, a_min=1, a_max=total_frames - 1).tolist()
        video_end = indexes[-self.action_chunk:]
        mem_candidates = indexes[:-self.action_chunk]
        if len(mem_candidates) < self.n_previous - 1:
            mem_candidates = [1, ] * (self.n_previous - 1) + mem_candidates

        if self.previous_pick_mode == 'uniform':
            mem_indexes = [mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates) - 1, self.n_previous).tolist()]
        elif self.previous_pick_mode == 'random':
            mem_indexes = [mem_candidates[i] for i in sorted(np.random.choice(list(range(0, len(mem_candidates) - 1)), size=self.n_previous - 1, replace=False).tolist())] + [mem_candidates[-1]]
        else:
            raise NotImplementedError(f"unsupported previous_pick_mode: {self.previous_pick_mode}")

        if not self.ignore_seek:
            frame_indexes = mem_indexes + video_end[self.video_temporal_stride - 1::self.video_temporal_stride]
        else:
            frame_indexes = mem_indexes + mem_indexes[-1:]

        action_indexes = mem_indexes + video_end
        return frame_indexes, action_indexes


    def get_action_bias_std(self, domain_name):
        return (
            torch.tensor(self.StatisticInfo[domain_name + "_" + self.action_space]['mean']).unsqueeze(0),
            torch.tensor(self.StatisticInfo[domain_name + "_" + self.action_space]['std']).unsqueeze(0) + 1e-6
        )


    def seek_mp4(self, video_path, cam_name_list, slices):
        video_list = []
        for cam_name in cam_name_list:
            video_reader = VideoFileClip(video_path.format(cam_name))
            fps = video_reader.fps
            video = []
            for idx in slices:
                video.append(video_reader.get_frame(float(idx) / fps))
            video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
            video = video.float() / 255.
            video_reader.close()
            video_list.append(video)
        return video_list


    def transform_video(self, videos, specific_transforms_resize, intrinsics, sample_size):
        v = len(videos)
        new_videos = []
        new_intrinsics = []
        for iv in range(v):
            video = videos[iv]
            c, t, h, w = video.shape
            if self.random_crop:
                h_start, w_start, h_crop, w_crop = gen_crop_config(video)
                video = video[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop]
                if intrinsics is not None:
                    intrinsic = intrin_crop_transform(intrinsics[iv], h_start, w_start)
                h, w = h_crop, w_crop
            if intrinsics is not None:
                intrinsic = intrinsic_transform(intrinsic, (h, w), sample_size, self.preprocess)
                new_intrinsics.append(intrinsic)
            video = specific_transforms_resize(video)
            new_videos.append(video)
        new_videos = torch.stack(new_videos, dim=1)
        if len(new_intrinsics) > 0:
            new_intrinsics = torch.stack(new_intrinsics, dim=0)
        else:
            new_intrinsics = None
        return new_videos, None


    def normalize_video(self, video, specific_transforms_norm):
        c, v, t, h, w = video.shape
        video = specific_transforms_norm(
            video.permute(1, 2, 0, 3, 4).reshape(-1, c, h, w)
        ).reshape(v, t, c, h, w).permute(2, 0, 1, 3, 4)
        return video


    def get_transform(self):
        return self.sample_size, self.pixel_transforms_resize, self.pixel_transforms_norm


    def get_long_recaption(self, step_captions, task_caption):
        newcap = []
        for step_caption in step_captions:
            if self.step_recap_map is not None:
                recap_list = self.step_recap_map.get(step_caption, [])
                recap_list.append(step_caption)
                step_caption = np.random.choice(recap_list, 1)
                newcap.append(str(step_caption[0]))
            else:
                newcap.append(step_caption)
        newcap = ", ".join(newcap)
        newcap = newcap.replace(" the ", " ")
        if self.task_recap_map is not None:
            task_recap_list = self.task_recap_map.get(task_caption, [])
            task_recap_list.append(task_caption)
            task_newcap = np.random.choice(task_recap_list, 1)
            task_newcap = str(task_newcap[0])
            fullcap = task_newcap + ": " + newcap
        else:
            task_newcap = task_caption
            fullcap = task_caption + ": " + newcap
        cap_type = random.randint(0, 2)
        allcap = [fullcap, task_newcap, newcap]
        return allcap[cap_type]


    def get_batch(self, idx):
        video_path = self.dataset[idx][0]
        parquet_path = self.dataset[idx][2]
        domain_name = self.dataset[idx][3]
        caption = self.dataset[idx][6]
        total_frames = self.dataset[idx][7]

        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()
        vid_indexes, indexes = self.get_frame_indexes(total_frames)
        data = pd.read_parquet(parquet_path)

        action_mean, action_std = self.get_action_bias_std(domain_name)
        state_mean, state_std = self.get_action_bias_std(domain_name + "_state")

        try:
            action = np.stack([data[self.action_key][i] for i in range(data[self.action_key].shape[0])])
            state = np.stack([data[self.state_key][i] for i in range(data[self.state_key].shape[0])])
        except:
            raise ValueError("We currently only support action and state data with the shape of T*C!")

        action = action.astype(np.float32)
        state = state.astype(np.float32)

        # ---- 保存原始 state（用于轨迹投影，不做归一化）----
        raw_state_full = state.copy()  # (T_total_episode, state_dim)

        state_for_return = torch.FloatTensor(state)[indexes][self.n_previous - 1:self.n_previous]
        state_for_return = (state_for_return - state_mean) / state_std

        if self.action_type == "absolute":
            action = action[indexes].astype(np.float32)
            action = torch.FloatTensor(action)
            action = (action - action_mean) / action_std
        elif self.action_type == "delta":
            delta_act_meanv, delta_act_stdv = self.get_action_bias_std(domain_name + "_delta")
            action_curr = torch.FloatTensor(action[indexes].astype(np.float32))
            action_last = torch.FloatTensor(action[[_ - 1 for _ in indexes]].astype(np.float32))
            delta_action = action_curr - action_last
            delta_action[:, 6] = action_last[:, 6]
            delta_action[:, 13] = action_last[:, 13]
            delta_action = (delta_action - delta_act_meanv) / delta_act_stdv
            action = delta_action
        elif self.action_type == "relative":
            action_curr = action[indexes].astype(np.float32)
            action = torch.FloatTensor(action_curr)
            action = (action - action_mean) / action_std
            rel_action = action - state_for_return
            rel_action[:, 6] = action[:, 6]
            rel_action[:, 13] = action[:, 13]
            action = rel_action
        else:
            raise NotImplementedError

        videos = self.seek_mp4(video_path, self.valid_cam, vid_indexes)
        videos, _ = self.transform_video(videos, specific_transforms_resize, None, sample_size)
        videos = self.normalize_video(videos, specific_transforms_norm)
        # videos shape: (C, V, T_mem+T_future, H, W), float, -1~1

        # ================================================================
        # 轨迹叠加逻辑
        # ================================================================
        video_with_traj = videos  # 默认 fallback

        if self.use_trajectory_condition:
            try:
                last_mem_global_idx = int(vid_indexes[self.n_previous - 1])
                raw_states_for_proj = raw_state_full

                mem_frames = videos[:, :, :self.n_previous]   # (C, V, T_mem, H, W)

                mem_with_traj = overlay_trajectory_on_video_tensor(
                    mem_video_tensor=mem_frames,
                    raw_states=raw_states_for_proj,
                    current_frame_idx=last_mem_global_idx,
                    draw_steps=self.trajectory_steps,
                    hfov=self.camera_hfov,
                    sensor_height=self.camera_sensor_height,
                    alpha=self.trajectory_alpha,
                )

                video_with_traj = videos.clone()
                video_with_traj[:, :, :self.n_previous] = mem_with_traj

                # ---- 调试保存 ----
                self._debug_counter += 1
                existing = len(glob.glob(os.path.join(self.trajectory_debug_dir, "*.jpg")))
                if self.trajectory_debug_dir is not None and existing < 10:  # 最多保存10组
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else 0
                    debug_img_path = os.path.join(
                        self.trajectory_debug_dir,
                        f"traj_debug_w{worker_id}_{self._debug_counter:06d}.jpg"
                    )
                    debug_vid_path = os.path.join(
                        self.trajectory_debug_dir,
                        f"traj_debug_w{worker_id}_{self._debug_counter:06d}.mp4"
                    )
                    # save_trajectory_debug_image(
                    #     frame_tensor=mem_with_traj[:, 0, -1],
                    #     raw_states=raw_states_for_proj,
                    #     current_frame_idx=last_mem_global_idx,
                    #     draw_steps=self.trajectory_steps,
                    #     save_path=debug_img_path,
                    #     hfov=self.camera_hfov,
                    #     sensor_height=self.camera_sensor_height,
                    # )
                    last_frame = mem_with_traj[:, 0, -1]  # (C, H, W)
                    frame_np = ((last_frame.float().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(debug_img_path, frame_bgr)
                    print(f"[DEBUG] 轨迹图保存至: {debug_img_path}")
                    save_trajectory_debug_video(
                        video_tensor=mem_with_traj,
                        save_path=debug_vid_path,
                        fps=10,
                        view_idx=0,
                    )

            except Exception as e:
                print(f"[TRAJ ERROR] {e}")
                import traceback
                traceback.print_exc()
                video_with_traj = videos  # 出错时用原始视频

        return videos, video_with_traj, action, caption, state_for_return


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        if self.fix_epiidx is not None:
            video, video_with_traj, actions, caption, state = self.get_batch(self.fix_epiidx)
        else:
            while True:
                try:
                    video, video_with_traj, actions, caption, state = self.get_batch(idx)
                    break
                except:
                    traceback.print_exc()
                    idx = random.randint(0, self.length - 1)

        sample = dict(
            video=video,                       # 原始视频，(C, V, T, H, W)
            video_with_traj=video_with_traj,   # mem帧叠加轨迹后的视频，shape 相同
            actions=actions,
            caption=caption,
            state=state,
        )
        return sample