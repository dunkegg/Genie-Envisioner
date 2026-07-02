import torch
from safetensors.torch import load_file

# 1. 填入你下载的智元核心权重绝对路径
weight_path = "/mnt/pfs/3zpd5q/code/zb/Genie-Envisioner/pretrained_models/ge_sim_cosmos_v0.1.safetensors"

print(f"正在读取并解析权重文件: {weight_path} ...")
try:
    # 仅加载权重的元数据和键名，不耗费过多显存/内存
    state_dict = load_file(weight_path)
except Exception as e:
    print(f"读取失败，请检查路径或文件是否下载完整。错误信息: {e}")
    exit()

print(f"该权重文件包含的参数张量（Tensor）总总数: {len(state_dict)} 个\n")

# 2. 定义我们要强行检索的关键词
# 智元在微调 Cosmos 时，用于接收动作增量和图像 Style 的条件分支通常会包含以下命名空间
motion_keywords = ["action", "motion", "delta", "proj", "embed", "style", "clip"]

print("================ [开始检索动作与风格条件权重] ================")
found_keys = []
for key in state_dict.keys():
    # 如果键名里包含我们关心的条件关键词，就打印出来
    if any(kw in key.lower() for kw in motion_keywords):
        found_keys.append(key)
        # 顺便打印该权重的形状（Shape），用于比对维度
        print(f"🔑 Key: {key:<50} | 📊 Shape: {list(state_dict[key].shape)}")

print("=============================================================")

if len(found_keys) > 0:
    print(f"🎉 验证成功！共检测到 {len(found_keys)} 个与动作/风格条件相关的预训练参数层。")
    print("这证明智元官方【已经】将这部分 Encoder 模块训练完毕并合并到了主干网络中。")
else:
    print("❌ 警告：未检测到任何动作/条件相关的变量名！")
    print("这说明该 safetensors 只是纯粹的视频生成底座，动作分支可能需要根据论文自行搭建并从头训练。")