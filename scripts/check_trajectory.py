import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

def visualize_trajectory_with_auto_rotation():
    # 1. 硬件参数与内参矩阵
    width, height = 720, 640
    hfov = 87
    sensor_height = 0.5
    
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    fy = fx
    cx, cy = width / 2.0, height / 2.0

    # 2. 读取数据 (请修改为你的实际路径)
    parquet_path = "lerobot_data/habitat_lerobot/data/chunk-000/episode_000000.parquet"
    video_path = "lerobot_data/habitat_lerobot/videos/chunk-000/observation.images.chest/episode_000000.mp4" 
    
    df = pd.read_parquet(parquet_path)
    clip = VideoFileClip(video_path)
    states = np.stack(df['observation.state'].values)
    
    # ==========================================
    # ⭐ 核心设置
    # 你可以把 current_frame_idx 改成 50 或者 100 试试看
    # 如果不用下面这段代码，在第 50 帧时旧代码的线大概率是歪的
    # ==========================================
    current_frame_idx = 0 
    draw_steps = 200
    
    # 取出当前帧画面
    frame_rgb = clip.get_frame(current_frame_idx / clip.fps)
    canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # 3. 提取当前帧 base 坐标
    base_x = states[current_frame_idx][0]
    base_z = states[current_frame_idx][1]
    
    # ==========================================
    # 🚀 绝处逢生的破局点：用运动矢量推导相机朝向
    # ==========================================
    # 取下一帧计算当前前进方向 (Forward Vector)
    target_next = min(current_frame_idx + 1, len(states) - 1)
    next_x = states[target_next][0]
    next_z = states[target_next][1]
    
    # 计算前进的向量 (dx, dz)
    fw_x = next_x - base_x
    fw_z = next_z - base_z
    length = np.sqrt(fw_x**2 + fw_z**2)
    
    if length > 1e-5:
        # 归一化得到正前方的单位向量
        f_x = fw_x / length
        f_z = fw_z / length
    else:
        # 如果没动，默认朝正前方 (Z轴正向)
        f_x, f_z = 0.0, 1.0
        
    # 推导正右方的单位向量 (Right Vector)
    # 假设 Z 是前进，X 是右边：当 forward 是 (0,1) 时，right 是 (1,0)
    r_x = f_z
    r_z = -f_x
    
    projected_points = []
    target_len = min(len(states), current_frame_idx + draw_steps + 1)
    
    # 4. 坐标投影循环
    for t in range(current_frame_idx + 1, target_len):
        # 算出绝对位置差 (上帝视角)
        world_dx = states[t][0] - base_x
        world_dz = states[t][1] - base_z
        
        # 🚀 将上帝视角的位移，点乘局部坐标系的向量，得到相机的真实局部坐标
        # 这就相当于把全局坐标系“拧”到了相机的正前方
        X_cam = world_dx * r_x + world_dz * r_z
        Z_cam = world_dx * f_x + world_dz * f_z
        Y_cam = sensor_height 
        
        if Z_cam <= 0.05:
            continue
            
        u = int((fx * X_cam / Z_cam) + cx)
        v = int((fy * Y_cam / Z_cam) + cy)
        
        # 为了看到长轨迹，允许超出屏幕外的点参与连线
        projected_points.append((u, v))
            
    # 5. 绘制轨迹
    for i in range(len(projected_points) - 1):
        pt1 = projected_points[i]
        pt2 = projected_points[i+1]
        
        ratio = i / len(projected_points)
        color = (0, int(255 * ratio), int(255 * (1 - ratio)))
        
        cv2.line(canvas, pt1, pt2, color, thickness=3, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt1, radius=4, color=color, thickness=-1)

    out_file = f"trajectory_auto_rot_{current_frame_idx}.jpg"
    cv2.imwrite(out_file, canvas)
    print(f"可视化完成！已保存为 {out_file}")

if __name__ == "__main__":
    visualize_trajectory_with_auto_rotation()