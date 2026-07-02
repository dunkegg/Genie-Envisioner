import os
import json
import glob
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import h5py


def print_hdf5_keys(h5_path):
    """
    递归打印 HDF5 文件中的所有 key / group / dataset
    """

    def _print_item(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name} shape={obj.shape} dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[GROUP]   {name}")

    with h5py.File(h5_path, "r") as f:
        print(f"\n===== HDF5 Structure: {h5_path} =====")
        f.visititems(_print_item)



# ==============================
# ⚙️ 配置
# ==============================

INPUT_GLOB = "data/h5_ed2/*.h5"   # 你的输入
OUTPUT_DIR = "data/lerobot/move_to_object"  # 输出目录
FPS = 30
CHUNK_SIZE = 1000  # 每多少个 episode 一个 chunk

USE_VELOCITY_AS_ACTION = True  # True: 用 vx,vy,vyaw；False: 用 delta state

STATE_NAMES = [
    "torso_roll",
    "torso_pitch",
    "torso_yaw",
    "body_height",
    "yaw_position",
    "linear_velocity_x",
    "linear_velocity_y",
    "angular_velocity_yaw",
]
ACTION_NAMES = [
    "torso_roll",
    "torso_pitch",
    "torso_yaw",
    "body_height",
    "yaw_position",
    "linear_velocity_x",
    "linear_velocity_y",
    "angular_velocity_yaw",
]
# ==============================
# 📊 统计函数（你之前问的）
# ==============================

def summarize_array(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


# ==============================
# 📂 创建目录
# ==============================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ==============================
# 📥 读取 HDF5
# ==============================
def load_hdf5(path):
    """
    读取 HDF5 中所有 traj group，
    返回 list[dict]
    """

    all_data = []

    with h5py.File(path, "r") as f:

        for traj_name in sorted(f.keys()):

            g = f[traj_name]

            # (T, 8)
            camera_pos = g["camera_pos"][:]

            data = {
                "torso_r": camera_pos[:, 0],
                "torso_p": camera_pos[:, 1],
                "torso_y": camera_pos[:, 2],
                "hb":      camera_pos[:, 3],
                "vx":      camera_pos[:, 4],
                "vy":      camera_pos[:, 5],
                "vyaw":    camera_pos[:, 6],
                "pyaw":    camera_pos[:, 7],
            }

            # instruction
            if "instruction" in g:
                instruction = g["instruction"][()]

                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")

                data["instruction"] = instruction

            # rgb
            if "rgb" in g:
                data["images"] = g["rgb"][:]

            all_data.append(data)

    return all_data

# def load_hdf5(path):
#     """
#     读取 HDF5 中的第一个 traj group，
#     并转换成原来的 Lerobot 格式字段
#     """

#     with h5py.File(path, "r") as f:

#         # 取第一个 trajectory group
#         first_traj = sorted(f.keys())[0]
#         g = f[first_traj]

#         # (T, 8)
#         camera_pos = g["camera_pos"][:]

#         # 你的 8 维定义：
#         # [
#         #   torso_r,
#         #   torso_p,
#         #   torso_y,
#         #   hb,
#         #   vx,
#         #   vy,
#         #   vyaw,
#         #   pyaw
#         # ]

#         data = {
#             "torso_r": camera_pos[:, 0],
#             "torso_p": camera_pos[:, 1],
#             "torso_y": camera_pos[:, 2],
#             "hb":      camera_pos[:, 3],
#             "vx":      camera_pos[:, 4],
#             "vy":      camera_pos[:, 5],
#             "vyaw":    camera_pos[:, 6],
#             "pyaw":    camera_pos[:, 7],
#         }

#         # instruction
#         if "instruction" in g:
#             instruction = g["instruction"][()]

#             # bytes -> str
#             if isinstance(instruction, bytes):
#                 instruction = instruction.decode("utf-8")

#             data["instruction"] = instruction

#         # rgb
#         if "rgb" in g:
#             data["images"] = g["rgb"][:]

#     return data

# def load_hdf5(path):
#     with h5py.File(path, "r") as f:

#         data = {
#             "torso_r": f["torso_r"][:],
#             "torso_p": f["torso_p"][:],
#             "torso_y": f["torso_y"][:],
#             "hb": f["hb"][:],
#             "vx": f["vx"][:],
#             "vy": f["vy"][:],
#             "vyaw": f["vyaw"][:],
#             "pyaw": f["pyaw"][:],
#         }

#         # 可选：图像
#         if "images" in f:
#             data["images"] = f["images"][:]

#     return data


# ==============================
# 🧠 构建 state / actions
# ==============================

def build_state_action(data):
    state = np.stack([
        data["torso_r"],
        data["torso_p"],
        data["torso_y"],
        data["hb"],
        data["pyaw"],
        data["vx"],
        data["vy"],
        data["vyaw"],
    ], axis=1)

    if USE_VELOCITY_AS_ACTION:
        actions = np.stack([
            data["torso_r"],
            data["torso_p"],
            data["torso_y"],
            data["hb"],
            data["pyaw"],
            data["vx"],
            data["vy"],
            data["vyaw"],
        ], axis=1)
    else:
        actions = state[1:] - state[:-1]
        state = state[:-1]

    return state, actions


# ==============================
# 🎥 写视频（可选）
# ==============================

def write_video(frames, path, fps=30):
    import cv2

    H, W = frames[0].shape[:2]

    ensure_dir(os.path.dirname(path))

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    for f in frames:
        writer.write(f)

    writer.release()

def compute_episode_stats(
    state,
    actions,
    ep_idx,
    fps,
):
    T = len(state)

    timestamp = np.arange(T) / fps
    frame_index = np.arange(T)
    episode_index_arr = np.full(T, ep_idx)
    index_arr = np.arange(T)
    task_index_arr = np.zeros(T)

    stats = {
        "timestamp": summarize_array(timestamp),
        "frame_index": summarize_array(frame_index),
        "episode_index": summarize_array(episode_index_arr),
        "index": summarize_array(index_arr),
        "task_index": summarize_array(task_index_arr),

        "observation.state": {
            **summarize_array(state),
            "dtype": "float32",
            "shape": [state.shape[1]],
            "names": STATE_NAMES,
        },

        "actions": {
            **summarize_array(actions),
            "dtype": "float32",
            "shape": [actions.shape[1]],
            "names": ACTION_NAMES,
        },
    }

    return {
        "episode_index": ep_idx,
        "stats": stats,
    }
# ==============================
# 🚀 主流程
# ==============================

def main():
    files = sorted(glob.glob(INPUT_GLOB))
    # file = files[0]
    print(f"Found {len(files)} files")

    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, "meta"))

    all_states = []
    all_actions = []

    episodes_meta = []
    episodes_stats = []
    tasks = []
    task_to_idx = {}
    next_task_idx = 0
    # for ep_idx, path in enumerate(tqdm(files)):
                # 用法
        # print_hdf5_keys(path)
    global_ep_idx = 0
    for file in files:
        all_data = load_hdf5(file)
        for k, data in enumerate(tqdm(all_data)):
            ep_idx = global_ep_idx
            global_ep_idx += 1
            # task_index = ep_idx
            state, actions = build_state_action(data)

            # instruction = data['instruction']
            instruction = data.get("instruction", "unknown task")

            if instruction not in task_to_idx:
                task_to_idx[instruction] = next_task_idx

                tasks.append({
                    "task_index": next_task_idx,
                    "task": instruction,
                })

                next_task_idx += 1

            task_index = task_to_idx[instruction]

            T = len(state)

            all_states.append(state)
            all_actions.append(actions)

            # chunk
            chunk_id = ep_idx // CHUNK_SIZE

            parquet_dir = os.path.join(
                OUTPUT_DIR,
                f"data/chunk-{chunk_id:03d}"
            )
            ensure_dir(parquet_dir)

            df = pd.DataFrame({
                # "observation.images.chest": list(data["images"]),  # ❗必须加
                # "observation.images.chest": [
                #     f"observation.images.chest/episode_{ep_idx:06d}.mp4"
                # ] * T,
                "observation.state": list(state),
                "actions": list(actions),
                "timestamp": np.arange(T) / FPS,
                "frame_index": np.arange(T),
                "episode_index": np.full(T, ep_idx),
                "task_index": np.full(T, task_index),   # ✅新增 为什么都是0 todo
            })

            parquet_path = os.path.join(
                parquet_dir,
                f"episode_{ep_idx:06d}.parquet"instruction
            )
            df.to_parquet(parquet_path)

            # video（如果有）
            # if "images" in data:
            video_dir = os.path.join(
                OUTPUT_DIR,
                f"videos/chunk-{chunk_id:03d}/observation.images.chest"
            )
            video_path = os.path.join(
                video_dir,
                f"episode_{ep_idx:06d}.mp4"
            )
            write_video(data["images"], video_path, FPS)

            # episode meta
            episodes_meta.append({
                "episode_index": ep_idx,
                "length": int(T),
                "tasks":[instruction]

            })

            
            episodes_stats.append(
                compute_episode_stats(state, actions, ep_idx, fps=FPS)
            )


    # ==============================
    # 📊 统计信息
    # ==============================

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    info = {
        "codebase_version": "v2.0",
        "robot_type": "humanoid_nav",

        "total_episodes": len(episodes_meta),

        "total_frames": int(sum(ep["length"] for ep in episodes_meta)),

        "total_tasks": 1,

        "chunks_size": CHUNK_SIZE,

        "fps": FPS,

        "splits": {
            "train": f"0:{len(episodes_meta)}"
        },

        "data_path":
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",

        "video_path":
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",

        "features": {

            # ======================
            # video
            # ======================

            "observation.images.chest": {
                "dtype": "video",

                "shape": [3, 640, 720],

                "names": [
                    "channels",
                    "height",
                    "width",
                ],

                "info": {
                    "video.height": 640,
                    "video.width": 720,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                }
            },

            # ======================
            # scalar fields
            # ======================

            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },

            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            # ======================
            # state
            # ======================

            "observation.state": {
                "dtype": "float32",

                "shape": [all_states.shape[1]],

                "names": STATE_NAMES,
            },

            # ======================
            # actions
            # ======================

            "actions": {
                "dtype": "float32",

                "shape": [all_actions.shape[1]],

                "names": ACTION_NAMES,
            },
        },

        "total_chunks":
            int(np.ceil(len(episodes_meta) / CHUNK_SIZE)),

        "total_videos":
            len(episodes_meta),
    }

    with open(os.path.join(OUTPUT_DIR, "meta/info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # ==============================
    # 📄 episodes.jsonl
    # ==============================

    with open(os.path.join(OUTPUT_DIR, "meta/episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    with open(os.path.join(OUTPUT_DIR, "meta/episodes_stats.jsonl"), "w") as f:
        for item in episodes_stats:
            f.write(json.dumps(item) + "\n")
    # ==============================
    # 📄 tasks.jsonl（简单版本）
    # ==============================

    with open(os.path.join(OUTPUT_DIR, "meta/tasks.jsonl"), "w") as f:
        for item in tasks:
            f.write(json.dumps(item) + "\n")

    # ==============================
    # 📄 tasks.jsonl（简单版本）
    # ==============================
    total_frames = all_states.shape[0]

    timestamp_arr = np.arange(total_frames) / FPS

    frame_index_arr = np.arange(total_frames)

    episode_index_arr = np.concatenate([
        np.full(ep["length"], ep["episode_index"])
        for ep in episodes_meta
    ])

    index_arr = np.arange(total_frames)

    task_index_arr = np.zeros(total_frames, dtype=np.int64)

    stats = {

        # ======================
        # scalar fields
        # ======================

        "timestamp": {
            **summarize_array(timestamp_arr),
            "dtype": "float32",
            "shape": [1],
            "names": None,
        },

        "frame_index": {
            **summarize_array(frame_index_arr),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },

        "episode_index": {
            **summarize_array(episode_index_arr),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },

        "index": {
            **summarize_array(index_arr),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },

        "task_index": {
            **summarize_array(task_index_arr),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },

        # ======================
        # state
        # ======================

        "observation.state": {
            **summarize_array(all_states),

            "dtype": "float32",

            "shape": [all_states.shape[1]],

            "names": STATE_NAMES,
        },

        # "observation.images.chest": {
        #     "min": [[[0.0]], [[0.0]], [[0.0]]],
        #     "max": [[[0.0]], [[0.0]], [[0.0]]],
        #     "mean": [[[0.0]], [[0.0]], [[0.0]]],
        #     "std": [[[1.0]], [[1.0]], [[1.0]]],

        #     "count": [total_frames],

        #     "q01": [[[0.0]], [[0.0]], [[0.0]]],
        #     "q10": [[[0.0]], [[0.0]], [[0.0]]],
        #     "q50": [[[0.0]], [[0.0]], [[0.0]]],
        #     "q90": [[[0.0]], [[0.0]], [[0.0]]],
        #     "q99": [[[0.0]], [[0.0]], [[0.0]]],

        #     "dtype": "video",

        #     "shape": [3, 640, 720],

        #     "names": [
        #         "channels",
        #         "height",
        #         "width",
        #     ],
        # },

        # ======================
        # actions
        # ======================

        "actions": {
            **summarize_array(all_actions),

            "dtype": "float32",

            "shape": [all_actions.shape[1]],

            "names": ACTION_NAMES,
        },
    }

    # stats = {
    #     "state": {
    #         **summarize_array(all_states),
    #         "dtype": "float32",
    #         "shape": [all_states.shape[1]],
    #         "names": STATE_NAMES,
    #     },
    #     "actions": {
    #         **summarize_array(all_actions),
    #         "dtype": "float32",
    #         "shape": [all_actions.shape[1]],
    #         "names": ACTION_NAMES,
    #     },
    # }

    with open(os.path.join(OUTPUT_DIR, "meta/stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()