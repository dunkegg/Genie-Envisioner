import os
import shutil
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

def transform_trajectory(trajectory, history_len, mode, scale_factor=2.0, mirror_axis=0):
    """
    对轨迹进行基于锚点的变换
    trajectory: [T, 2] 的 NumPy 矩阵
    """
    new_traj = trajectory.copy()
    
    # 如果总长度还没历史帧长，说明没有未来动作，直接返回
    if len(trajectory) <= history_len:
        return new_traj
        
    # 锚点：历史轨迹的最后一帧坐标
    anchor = trajectory[history_len - 1]
    
    # 提取未来轨迹
    future_traj = trajectory[history_len:]
    
    # 计算未来轨迹相对于锚点的相对位移 (Delta)
    deltas = future_traj - anchor
    
    if mode == 'scale':
        # 相对位移成比例放大
        new_deltas = deltas * scale_factor
    elif mode == 'mirror':
        # 仅将指定轴（比如 X 轴）的相对位移取反，实现镜像转弯
        new_deltas = deltas.copy()
        new_deltas[:, mirror_axis] = -new_deltas[:, mirror_axis]
        
    # 将变换后的位移加回锚点，得到新的绝对坐标
    new_traj[history_len:] = anchor + new_deltas
    
    return new_traj

def process_and_save_dataset(src_root, dst_root, history_len, mode='scale', scale_factor=2.5, mirror_axis=0):
    print(f"\n 开始生成 [{mode}] 数据集，保存至: {dst_root}")
    print(f"   - 历史保留长度: 前 {history_len} 帧保持原样")
    
    os.makedirs(dst_root, exist_ok=True)
    
    # 软链接 videos 和 meta
    for folder in ['videos', 'meta']:
        src_folder = os.path.join(src_root, folder)
        dst_folder = os.path.join(dst_root, folder)
        if os.path.exists(src_folder) and not os.path.exists(dst_folder):
            os.symlink(os.path.abspath(src_folder), dst_folder)
            
    # 复制 statistics.json
    stat_file = 'statistics.json'
    if os.path.exists(os.path.join(src_root, stat_file)):
        shutil.copy(os.path.join(src_root, stat_file), os.path.join(dst_root, stat_file))

    parquet_files = glob.glob(os.path.join(src_root, 'data', 'chunk-*', '*.parquet'))
    
    for src_parquet in tqdm(parquet_files, desc=f"Processing {mode} parquets"):
        rel_path = os.path.relpath(src_parquet, src_root)
        dst_parquet = os.path.join(dst_root, rel_path)
        os.makedirs(os.path.dirname(dst_parquet), exist_ok=True)
        
        df = pd.read_parquet(src_parquet)
        
        # [T, 2]
        actions = np.stack(df['actions'].values).astype(np.float32)
        states = np.stack(df['observation.state'].values).astype(np.float32)
        
        # 核心：执行基于锚点的轨迹变换
        actions_transformed = transform_trajectory(actions, history_len, mode, scale_factor, mirror_axis)
        states_transformed = transform_trajectory(states, history_len, mode, scale_factor, mirror_axis)

        # 转换回列表存入 DataFrame
        df['actions'] = list(actions_transformed)
        df['observation.state'] = list(states_transformed)
        
        df.to_parquet(dst_parquet, index=False)

if __name__ == "__main__":
    DATASET_ROOT = "lerobot_data/habitat_lerobot"
    SCALED_ROOT = "lerobot_data/habitat_lerobot_scaled"
    MIRRORED_ROOT = "lerobot_data/habitat_lerobot_mirrored"
    
    # 这里必须改成你 yaml 配置文件中的 n_previous 的值！
    # 如果你历史帧输入 16 帧，这里就填 16
    HISTORY_LEN = 8
    
    # 1. 生成 Scale (放大) 数据集
    process_and_save_dataset(
        src_root=DATASET_ROOT, 
        dst_root=SCALED_ROOT, 
        history_len=HISTORY_LEN,
        mode='scale', 
        scale_factor=2.5  # 放大倍数
    )
    
    # 2. 生成 Mirror (镜像相反方向) 数据集
    process_and_save_dataset(
        src_root=DATASET_ROOT, 
        dst_root=MIRRORED_ROOT, 
        history_len=HISTORY_LEN,
        mode='mirror', 
        mirror_axis=0     # 假设 0 是左右横向（X轴），取反实现反向转弯
    )
    
    print("\n 所有扰动数据集生成完毕！物理轨迹衔接完美！")