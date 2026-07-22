import argparse
import logging
import os
import socket
import sys

import torch

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root_dir)

from web_infer_utils.za_adaptor_server import ZaAdaptorServer


def parse_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def get_args():
    parser = argparse.ArgumentParser(description="Serve TD-MPC2 actions from GE Za-adaptor external z over websocket.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to Za adaptor server YAML.")
    parser.add_argument(
        "-w",
        "--adaptor_checkpoint",
        type=str,
        default=None,
        help="Path to step_x/za_adaptor.pt. If omitted, read za_adaptor.checkpoint_path from YAML.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("-p", "--port", type=int, default=8011)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--n_prev", type=int, default=None)
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--tdmpc_project_root", type=str, default=None, help="Directory containing exp_config.py and tdmpc2.py.")
    parser.add_argument("--tdmpc_model_path", type=str, default=None, help="TD-MPC2 checkpoint path.")
    parser.add_argument("--tdmpc_cfg_args", type=str, default=None, help="Extra args passed to TD-MPC2 exp_config.get_config().")
    parser.add_argument("--tdmpc_gpu", type=int, default=None, help="CUDA GPU id passed to TD-MPC2.")
    parser.add_argument("--action_format", type=str, default=None, choices=["real", "norm"], help="Action format returned as response['actions'].")
    parser.add_argument(
        "--disable_online_training",
        action="store_true",
        help="Serve inference only even if tdmpc.online_training.enabled is true in the YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    metadata = {
        "name": "Genie-Envisioner Za adaptor + TD-MPC2 action server",
        "output": "actions",
        "z_dim": 512,
        "request_schema": "obs + prompt + execution_step + beta OR robot_state + goal_rel; optional reward/done for online TD-MPC training; returns TD-MPC2 action",
    }
    tdmpc_overrides = {
        "project_root": args.tdmpc_project_root,
        "model_path": args.tdmpc_model_path,
        "tdmpc_cfg_args": args.tdmpc_cfg_args,
        "gpu": args.tdmpc_gpu,
        "action_format": args.action_format,
        "online_training": {"enabled": False} if args.disable_online_training else None,
    }
    actor = ZaAdaptorServer(
        args.host,
        args.port,
        metadata,
        tdmpc_overrides=tdmpc_overrides,
        config_file=args.config,
        adaptor_checkpoint=args.adaptor_checkpoint,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        n_prev=args.n_prev,
        threshold=args.threshold,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating Za adaptor server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)
    print("Waiting...")
    actor.serve_forever()
