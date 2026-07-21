"""TD-MPC2 bridge adapter for the existing Isaac random navigation scene.

The public class in this module follows scripts/tdmpc_bridge/
README_Isaac_Env_Adapter.md:

    obs = {
        "laser_map": np.ndarray,  # [128, 256], float32, 1=free, 0=obstacle
        "velocity": np.ndarray,   # [v, w], real m/s and rad/s
        "goal_rel": np.ndarray,   # [distance, theta], theta>0 is robot-left
    }

The current Isaac scene in this repository contains one H1 robot.  Therefore
this adapter intentionally supports one agent.  Start the z dataset collector
with --num-agents 1 and {"num_agents": 1} in --isaac-env-kwargs-json.
"""

from __future__ import annotations
import sys
from pathlib import Path

def add_project_root_to_sys_path():
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "nav_isaaclab").is_dir():
            project_root = parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            print(f"[path] added project root: {project_root}", flush=True)
            return project_root

    raise RuntimeError(
        f"Cannot find project root containing nav_isaaclab from: {current}"
    )

PROJECT_ROOT = add_project_root_to_sys_path()

import nav_isaaclab.assets.sim_env as sim_env
import argparse
import math
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _normalize_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _quat_wxyz_to_yaw(quat_wxyz: Any) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if q.size < 4:
        return 0.0
    w, x, y, z = [float(v) for v in q[:4]]
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _is_point_in_blocked_region(point_xy, region, margin=0.0):
    if region.get("type") == "rect":
        center = region.get("center", [0.0, 0.0])
        size = region.get("size", [0.0, 0.0])
        return (
            abs(float(point_xy[0]) - float(center[0])) <= float(size[0]) / 2.0 + float(margin)
            and abs(float(point_xy[1]) - float(center[1])) <= float(size[1]) / 2.0 + float(margin)
        )
    if region.get("type") == "circle":
        center = region.get("center", [0.0, 0.0])
        radius = float(region.get("radius", 0.0)) + float(margin)
        dx = float(point_xy[0]) - float(center[0])
        dy = float(point_xy[1]) - float(center[1])
        return dx * dx + dy * dy <= radius * radius
    return False


def _check_layout_collision(robot_xy, layout_metadata, robot_radius=0.35):
    bounds = (layout_metadata or {}).get("factory_bounds")
    if bounds and len(bounds) >= 4:
        x_min, x_max, y_min, y_max = [float(v) for v in bounds[:4]]
        if (
            float(robot_xy[0]) < x_min + robot_radius
            or float(robot_xy[0]) > x_max - robot_radius
            or float(robot_xy[1]) < y_min + robot_radius
            or float(robot_xy[1]) > y_max - robot_radius
        ):
            return True, "factory_wall"

    for region in (layout_metadata or {}).get("blocked_regions", []):
        label = str(region.get("label", "object"))
        if _is_point_in_blocked_region(robot_xy, region, margin=robot_radius):
            return True, label
    return False, ""


def _is_done_signal(done) -> bool:
    if done is None:
        return False
    if isinstance(done, (bool, np.bool_)):
        return bool(done)
    if hasattr(done, "detach"):
        return bool(done.detach().cpu().any().item())
    if isinstance(done, np.ndarray):
        return bool(np.any(done))
    try:
        return bool(done)
    except Exception:
        return False


def _sample_random_factory_target(layout_metadata, rng, robot_xy=None, min_robot_distance=1.5):
    bounds = (layout_metadata or {}).get("factory_bounds", [-10.0, 10.0, -15.0, 15.0])
    x_min, x_max, y_min, y_max = [float(v) for v in bounds[:4]]
    blocked_regions = (layout_metadata or {}).get("blocked_regions", [])
    for _ in range(300):
        x = float(rng.uniform(x_min + 0.75, x_max - 0.75))
        y = float(rng.uniform(y_min + 0.75, y_max - 0.75))
        candidate = np.array([x, y], dtype=np.float32)
        if robot_xy is not None:
            robot_xy_arr = np.asarray(robot_xy[:2], dtype=np.float32)
            if float(np.linalg.norm(candidate - robot_xy_arr)) < float(min_robot_distance):
                continue
        if any(_is_point_in_blocked_region(candidate, region, margin=0.55) for region in blocked_regions):
            continue
        yaw = 0.0
        if robot_xy is not None:
            delta = candidate - np.asarray(robot_xy[:2], dtype=np.float32)
            yaw = math.atan2(float(delta[1]), float(delta[0]))
        return np.array([x, y, yaw], dtype=np.float32)

    return np.array([(x_min + x_max) / 2.0, 0.0, 0.0], dtype=np.float32)


def _sample_random_factory_route(layout_metadata, rng, robot_xy, waypoint_min=10, waypoint_max=10):
    robot_xy = np.asarray(robot_xy[:2], dtype=np.float32)
    waypoint_count = int(rng.integers(waypoint_min, waypoint_max + 1))
    final_target = _sample_random_factory_target(
        layout_metadata,
        rng,
        robot_xy=robot_xy,
        min_robot_distance=6.0,
    )[:2]

    delta = np.asarray(final_target, dtype=np.float32) - robot_xy
    dist = max(float(np.linalg.norm(delta)), 1e-6)
    direction = delta / dist
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    bend = float(rng.uniform(-1.0, 1.0))
    if abs(bend) < 0.25:
        bend = 0.45 if bend >= 0.0 else -0.45
    bend_scale = min(max(dist * 0.35, 2.5), 8.0) * bend

    control_1 = robot_xy + delta * float(rng.uniform(0.25, 0.4)) + normal * bend_scale
    control_2 = robot_xy + delta * float(rng.uniform(0.6, 0.8)) - normal * bend_scale * float(rng.uniform(0.35, 0.85))
    blocked_regions = (layout_metadata or {}).get("blocked_regions", [])
    bounds = (layout_metadata or {}).get("factory_bounds", [-10.0, 10.0, -15.0, 15.0])
    x_min, x_max, y_min, y_max = [float(v) for v in bounds[:4]]

    points = []
    for t in np.linspace(1.0 / waypoint_count, 1.0, waypoint_count):
        t = float(t)
        p = (
            (1.0 - t) ** 3 * robot_xy
            + 3.0 * (1.0 - t) ** 2 * t * control_1
            + 3.0 * (1.0 - t) * t ** 2 * control_2
            + t ** 3 * final_target
        )
        p[0] = float(np.clip(p[0], x_min + 0.8, x_max - 0.8))
        p[1] = float(np.clip(p[1], y_min + 0.8, y_max - 0.8))

        if any(_is_point_in_blocked_region(p, region, margin=0.6) for region in blocked_regions):
            for _ in range(40):
                candidate = p + rng.normal(0.0, 1.2, size=2).astype(np.float32)
                candidate[0] = float(np.clip(candidate[0], x_min + 0.8, x_max - 0.8))
                candidate[1] = float(np.clip(candidate[1], y_min + 0.8, y_max - 0.8))
                if not any(_is_point_in_blocked_region(candidate, region, margin=0.6) for region in blocked_regions):
                    p = candidate
                    break

        points.append(np.asarray(p, dtype=np.float32))

    waypoints = []
    for idx, point in enumerate(points):
        if idx == len(points) - 1 and len(points) >= 2:
            ref = points[idx - 1]
            yaw = math.atan2(float(point[1] - ref[1]), float(point[0] - ref[0]))
        else:
            ref = points[min(idx + 1, len(points) - 1)]
            yaw = math.atan2(float(ref[1] - point[1]), float(ref[0] - point[0]))
        waypoints.append(np.array([float(point[0]), float(point[1]), float(yaw)], dtype=np.float32))
    return waypoints


def _world_to_robot_frame_xy(xy, robot_xy, robot_yaw):
    dx = float(xy[0]) - float(robot_xy[0])
    dy = float(xy[1]) - float(robot_xy[1])
    cos_yaw = math.cos(float(robot_yaw))
    sin_yaw = math.sin(float(robot_yaw))
    forward = cos_yaw * dx + sin_yaw * dy
    left = -sin_yaw * dx + cos_yaw * dy
    return forward, left


def _robot_frame_to_bev_pixel(forward, left, resolution, map_height, map_width):
    if float(forward) < 0.0:
        return None
    bottom_row = int(map_height) - 1
    row = int(round(bottom_row - float(forward) / float(resolution)))
    if row < 0 or row >= int(map_height):
        return None
    center = (int(map_width) - 1) / 2.0
    col = int(round(center - float(left) / float(resolution)))
    if col < 0 or col >= int(map_width):
        return None
    return row, col


def _paint_disk_robot_frame(bev_map, center_xy, robot_xy, robot_yaw, resolution, radius, value):
    map_height, map_width = bev_map.shape[:2]
    forward, left = _world_to_robot_frame_xy(center_xy, robot_xy, robot_yaw)
    pixel = _robot_frame_to_bev_pixel(forward, left, resolution, map_height, map_width)
    if pixel is None:
        return
    center_row, center_col = pixel
    radius_px = max(1, int(math.ceil(float(radius) / float(resolution))))
    for row in range(max(0, center_row - radius_px), min(map_height, center_row + radius_px + 1)):
        for col in range(max(0, center_col - radius_px), min(map_width, center_col + radius_px + 1)):
            dx = (col - center_col) * float(resolution)
            dy = (row - center_row) * float(resolution)
            if dx * dx + dy * dy <= float(radius) * float(radius):
                bev_map[row, col] = int(value)


def _paint_rect_robot_frame(bev_map, center_xy, size_xy, robot_xy, robot_yaw, resolution, value=0):
    half_x = float(size_xy[0]) / 2.0
    half_y = float(size_xy[1]) / 2.0
    step = max(float(resolution) * 0.5, 0.04)
    xs = np.arange(float(center_xy[0]) - half_x, float(center_xy[0]) + half_x + step, step)
    ys = np.arange(float(center_xy[1]) - half_y, float(center_xy[1]) + half_y + step, step)
    for x in xs:
        for y in ys:
            _paint_disk_robot_frame(bev_map, (x, y), robot_xy, robot_yaw, resolution, step * 0.75, value)


def _build_local_laser_map(
    robot_xy,
    robot_yaw,
    layout_metadata,
    map_height=128,
    map_width=256,
    resolution=0.0625,
):
    """Build a local first-person occupancy image. Output: float32, 1=free, 0=obstacle."""
    bev_map = np.full((int(map_height), int(map_width)), 255, dtype=np.uint8)
    robot_xy = np.asarray(robot_xy[:2], dtype=np.float32)
    for region in (layout_metadata or {}).get("blocked_regions", []):
        if region.get("type") == "rect":
            _paint_rect_robot_frame(
                bev_map,
                region.get("center", [0.0, 0.0]),
                region.get("size", [0.0, 0.0]),
                robot_xy,
                robot_yaw,
                resolution,
                value=0,
            )
        elif region.get("type") == "circle":
            _paint_disk_robot_frame(
                bev_map,
                region.get("center", [0.0, 0.0]),
                robot_xy,
                robot_yaw,
                resolution,
                float(region.get("radius", 0.0)) + 0.08,
                value=0,
            )
    return (bev_map.astype(np.float32) / 255.0).astype(np.float32)


@dataclass
class _IsaacArgs:
    seed: int | None = None
    resume: bool | None = None
    load_run: str | None = "2024-11-03_15-08-09_height_scan_obst"
    checkpoint: str | None = None
    save_interval: int | None = None
    run_name: str | None = None
    logger: str | None = None
    log_project_name: str | None = None
    use_cnn: bool | None = None
    use_rnn: bool = False


class IsaacRandomNavEnv:
    """Single-H1 random navigation adapter for tdmpc_bridge z collection."""

    def __init__(
        self,
        num_envs=1,
        num_agents=1,
        headless=True,
        task="h1_with_lidar_camera_vision",
        env_name="create_table_and_factory_env",
        seed=None,
        action_format="real",
        max_linear_vel=4.0,
        max_angular_vel=3.0,
        map_height=128,
        map_width=256,
        map_resolution=0.0625,
        obstacle_count=20,
        target_reached_threshold=0.4,
        done_on_arrive=True,
        done_on_collision=True,
        route_waypoint_count=10,
        launch_app=True,
        load_run="2024-11-03_15-08-09_height_scan_obst",
        checkpoint=None,
        use_cnn=None,
        use_rnn=False,
        auto_reset=True,
        **kwargs,
    ):
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        if self.num_envs != 1:
            raise ValueError("This adapter currently supports num_envs=1 for the single-H1 Isaac scene.")
        if self.num_agents != 1:
            raise ValueError(
                "This adapter currently supports num_agents=1. "
                "Run the collector with --num-agents 1 and {'num_agents': 1}."
            )

        self.headless = bool(headless)
        self.task = str(task)
        self.env_name = str(env_name)
        self.action_format = str(action_format)
        self.max_linear_vel = float(max_linear_vel)
        self.max_angular_vel = float(max_angular_vel)
        self.map_height = int(map_height)
        self.map_width = int(map_width)
        self.map_resolution = float(map_resolution)
        self.obstacle_count = int(obstacle_count)
        self.target_reached_threshold = float(target_reached_threshold)
        self.done_on_arrive = bool(done_on_arrive)
        self.done_on_collision = bool(done_on_collision)
        self.route_waypoint_count = int(route_waypoint_count)
        self.rng = np.random.default_rng(0 if seed is None else int(seed))
        self.dynamic_layout_generation = 0
        self.current_target_idx = 0
        self.target_points = []
        self.layout_metadata = {}
        self.last_action = np.zeros(2, dtype=np.float32)
        self.latest_rgb = None

        self._isaac_args = _IsaacArgs(
            seed=seed,
            load_run=load_run,
            checkpoint=checkpoint,
            use_cnn=use_cnn,
            use_rnn=bool(use_rnn),
        )
        self._app_launcher = None
        self._simulation_app = None

        if launch_app:
            self._launch_isaac_app()
        self._init_isaac_env()
        if auto_reset:
            self.reset(seed=seed)
        else:
            self.initialize_without_warmup(seed=seed)

    def _launch_isaac_app(self):
        try:
            from isaaclab.app import AppLauncher
        except Exception as exc:
            raise RuntimeError(
                "Could not import isaaclab.app.AppLauncher. Run this adapter with Isaac Lab "
                "(for example: isaaclab.sh -p ...), or set launch_app=False if the app is already running."
            ) from exc

        parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(parser)
        args, _ = parser.parse_known_args([])
        if hasattr(args, "headless"):
            args.headless = self.headless
        if hasattr(args, "enable_cameras"):
            args.enable_cameras = True
        try:
            self._app_launcher = AppLauncher(args)
            self._simulation_app = self._app_launcher.app
        except Exception as exc:
            # Some collectors may already have launched Kit.  In that case keep
            # going and let gym.make verify that an app is usable.
            print(f"[IsaacRandomNavEnv] AppLauncher skipped/failed: {exc}", flush=True)

    def _init_isaac_env(self):
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import nav_isaaclab.assets.sim_env as sim_env
        import nav_isaaclab.tasks  # noqa: F401
        from nav_isaaclab.utils import EnvRLWrapper

        self._torch = torch
        self._sim_env = sim_env

        env_cfg = parse_env_cfg(self.task, num_envs=self.num_envs)
        env_cfg.scene.robot.init_state.pos = (-8.0, 0.0, 0.17162801325321198)
        yaw_offset = -0.1232834630188942
        initial_yaw = 0.0 - yaw_offset
        env_cfg.scene.robot.init_state.rot = (
            math.cos(initial_yaw / 2.0),
            0.0,
            0.0,
            math.sin(initial_yaw / 2.0),
        )

        base_env = RslRlVecEnvWrapper(gym.make(self.task, cfg=env_cfg, render_mode=None))
        args_cli = SimpleNamespace(**self._isaac_args.__dict__)
        self.env = EnvRLWrapper(base_env, self.task, args_cli=args_cli, high_level_obs_key="camera_obs")

        if self.env_name != "create_table_and_factory_env":
            raise ValueError("IsaacRandomNavEnv currently expects env_name='create_table_and_factory_env'.")
        self.layout_metadata = self._sim_env.create_table_and_factory_env() or {}
        if not self.layout_metadata and hasattr(self._sim_env, "get_table_factory_layout_metadata"):
            self.layout_metadata = self._sim_env.get_table_factory_layout_metadata() or {}

    def _extract_rgb(self, camera_obs):
        if camera_obs is None:
            return None
        if isinstance(camera_obs, dict):
            for key in ("rgb", "image", "camera_rgb", "camera"):
                if key in camera_obs:
                    rgb = self._extract_rgb(camera_obs[key])
                    if rgb is not None:
                        return rgb
            return None
        if hasattr(camera_obs, "detach"):
            arr = camera_obs.detach().cpu().numpy()
        else:
            arr = np.asarray(camera_obs)
        if arr.size == 0:
            return None
        arr = np.squeeze(arr)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr[..., :3]
        elif arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = np.transpose(arr[:3], (1, 2, 0))
        else:
            return None
        if np.issubdtype(arr.dtype, np.floating):
            max_value = float(np.nanmax(arr)) if arr.size else 1.0
            if max_value <= 1.0:
                arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8, copy=False)

    def _update_latest_rgb(self, camera_obs):
        rgb = self._extract_rgb(camera_obs)
        if rgb is not None:
            self.latest_rgb = rgb

    def _update_latest_rgb_from_infos(self, infos):
        try:
            observations = infos.get("observations", {})
            self._update_latest_rgb(observations.get(self.env.high_level_obs_key))
        except Exception:
            pass

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        camera_obs, _ = self.env.reset()
        self._update_latest_rgb(camera_obs)
        self.current_target_idx = 0
        self._refresh_route_and_layout(reset_robot=False)
        self.last_action = np.zeros(2, dtype=np.float32)
        return self._get_obs_all()

    def initialize_without_warmup(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        low_level_obs, infos = self.env.env.reset()
        self._update_latest_rgb_from_infos(infos)
        self.env.low_level_obs = low_level_obs
        zero_cmd = self._torch.zeros(3, device=low_level_obs.device, dtype=self._torch.float32)
        self.env.update_command(zero_cmd)
        self.env.low_level_action = None
        self.env.env_step = 0
        self.env.same_pos_count = 0
        self.env.prev_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach()
        self.current_target_idx = 0
        self._refresh_route_and_layout(reset_robot=False)
        self.last_action = np.zeros(2, dtype=np.float32)
        return self._get_obs_all()

    def observe(self):
        return self._get_obs_all()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.shape[0] < 1 or action.shape[1] < 2:
            raise ValueError(f"action must have shape [num_agents, 2], got {action.shape}")

        v, w = self._convert_action(action[0])
        self.last_action = np.array([v, w], dtype=np.float32)
        cmd = self._torch.tensor([v, 0.0, w], device=self.env.unwrapped.device, dtype=self._torch.float32)
        camera_obs, _, env_done, info = self.env.step(cmd)
        self._update_latest_rgb(camera_obs)

        robot_pos, robot_yaw = self._get_robot_pose()
        robot_xy = robot_pos[:2]
        collided, collision_label = _check_layout_collision(robot_xy, self.layout_metadata)
        env_done = _is_done_signal(env_done)
        arrive = self._advance_or_refresh_route(robot_pos, robot_yaw) if not (collided or env_done) else False

        done = False
        reset_reason = ""
        if collided or env_done:
            reset_reason = collision_label or "env_done"
            done = self.done_on_collision
            self.initialize_without_warmup()
            self.current_target_idx = 0
        elif arrive:
            reset_reason = "reach_goal"
            done = self.done_on_arrive

        obs = self._get_obs_all()
        reward = np.zeros(self.num_agents, dtype=np.float32)
        done_arr = np.asarray([done], dtype=bool)
        info = {
            "collision": bool(collided or env_done),
            "collision_label": reset_reason if (collided or env_done) else "",
            "arrive": bool(arrive),
            "target_index": int(self.current_target_idx),
            "num_targets": int(len(self.target_points)),
            "layout_seed": int(self.dynamic_layout_generation - 1),
            "raw_env_info": info,
        }
        return obs, reward, done_arr, info

    def close(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
        if self._simulation_app is not None:
            try:
                self._simulation_app.close()
            except Exception:
                pass

    def _convert_action(self, action_2d):
        v = float(action_2d[0])
        w = float(action_2d[1])
        if self.action_format == "norm":
            v = ((v + 1.0) / 2.0) * self.max_linear_vel
            w = w * self.max_angular_vel
        elif self.action_format != "real":
            raise ValueError(f"Unsupported action_format={self.action_format!r}; expected 'real' or 'norm'.")
        v = float(np.clip(v, 0.0, self.max_linear_vel))
        w = float(np.clip(w, -self.max_angular_vel, self.max_angular_vel))
        return v, w

    def compute_route_action(self):
        robot_pos, robot_yaw = self._get_robot_pose()
        if not self.target_points:
            self._refresh_route_and_layout(reset_robot=False)
        target = self.target_points[min(self.current_target_idx, len(self.target_points) - 1)]
        dx = float(target[0]) - float(robot_pos[0])
        dy = float(target[1]) - float(robot_pos[1])
        distance = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        heading_error = _normalize_angle(heading - float(robot_yaw))

        v = min(self.max_linear_vel, max(0.15, 0.9 * distance))
        if abs(heading_error) > 0.8:
            v *= 0.25
        elif abs(heading_error) > 0.45:
            v *= 0.5
        w = float(np.clip(1.8 * heading_error, -self.max_angular_vel, self.max_angular_vel))
        return np.asarray([v, w], dtype=np.float32)

    def _refresh_route_and_layout(self, reset_robot=False):
        robot_pos, _ = self._get_robot_pose()
        self.target_points = _sample_random_factory_route(
            self.layout_metadata,
            self.rng,
            robot_xy=robot_pos[:2],
            waypoint_min=self.route_waypoint_count,
            waypoint_max=self.route_waypoint_count,
        )
        self.layout_metadata = self._sim_env.regenerate_table_factory_dynamic_layout(
            route_points=self.target_points,
            seed=self.dynamic_layout_generation,
            obstacle_count=self.obstacle_count,
        )
        self.dynamic_layout_generation += 1
        if reset_robot:
            self.env.reset()

    def _advance_or_refresh_route(self, robot_pos, robot_yaw):
        if not self.target_points:
            self._refresh_route_and_layout(reset_robot=False)
            return False
        target = self.target_points[min(self.current_target_idx, len(self.target_points) - 1)]
        distance = float(np.linalg.norm(np.asarray(target[:2], dtype=np.float32) - np.asarray(robot_pos[:2], dtype=np.float32)))
        if distance > self.target_reached_threshold:
            return False
        if self.current_target_idx < len(self.target_points) - 1:
            self.current_target_idx += 1
            return False
        self.current_target_idx = 0
        self._refresh_route_and_layout(reset_robot=False)
        return True

    def _get_obs_all(self):
        return [self._get_obs_one(0)]

    def _get_obs_one(self, i):
        robot_pos, robot_yaw = self._get_robot_pose()
        laser_map = _build_local_laser_map(
            robot_pos[:2],
            robot_yaw,
            self.layout_metadata,
            map_height=self.map_height,
            map_width=self.map_width,
            resolution=self.map_resolution,
        )
        velocity = self._get_robot_velocity(robot_yaw)
        goal_rel = self._get_goal_relative_polar(robot_pos, robot_yaw)
        target = self._current_target_xy()
        collided, collision_label = _check_layout_collision(robot_pos[:2], self.layout_metadata)
        obs = {
            "laser_map": laser_map,
            "velocity": velocity.astype(np.float32),
            "goal_rel": goal_rel.astype(np.float32),
            "pose": np.array([float(robot_pos[0]), float(robot_pos[1]), float(robot_yaw)], dtype=np.float32),
            "goal": target.astype(np.float32),
            "collision": bool(collided),
            "collision_label": collision_label,
            "arrive": bool(goal_rel[0] <= self.target_reached_threshold),
            "robot_id": int(i),
            "min_dist": float(self._estimate_min_dist(robot_pos[:2])),
        }
        if self.latest_rgb is not None:
            obs["rgb"] = self.latest_rgb
        return obs

    def _get_robot_pose(self):
        robot = self.env.unwrapped.scene["robot"]
        pos = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        yaw = _quat_wxyz_to_yaw(quat)
        return pos, yaw

    def _get_robot_velocity(self, robot_yaw):
        robot = self.env.unwrapped.scene["robot"]
        try:
            lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().numpy()
            ang_vel = robot.data.root_ang_vel_w[0].detach().cpu().numpy()
            v = math.cos(float(robot_yaw)) * float(lin_vel[0]) + math.sin(float(robot_yaw)) * float(lin_vel[1])
            w = float(ang_vel[2])
            return np.array([v, w], dtype=np.float32)
        except Exception:
            return self.last_action.astype(np.float32)

    def _get_goal_relative_polar(self, robot_pos, robot_yaw):
        if not self.target_points:
            return np.zeros(2, dtype=np.float32)
        target_xy = self._current_target_xy()
        robot_xy = np.asarray(robot_pos[:2], dtype=np.float32)
        delta = target_xy - robot_xy
        rho = float(np.linalg.norm(delta))
        world_bearing = math.atan2(float(delta[1]), float(delta[0]))
        theta = _normalize_angle(world_bearing - float(robot_yaw))
        return np.array([rho, theta], dtype=np.float32)

    def _current_target_xy(self):
        if not self.target_points:
            return np.zeros(2, dtype=np.float32)
        idx = min(max(int(self.current_target_idx), 0), len(self.target_points) - 1)
        return np.asarray(self.target_points[idx][:2], dtype=np.float32)

    def _estimate_min_dist(self, robot_xy):
        best = float("inf")
        for region in (self.layout_metadata or {}).get("blocked_regions", []):
            if region.get("type") == "circle":
                center = np.asarray(region.get("center", [0.0, 0.0]), dtype=np.float32)
                best = min(best, float(np.linalg.norm(np.asarray(robot_xy, dtype=np.float32) - center) - float(region.get("radius", 0.0))))
            elif region.get("type") == "rect":
                center = np.asarray(region.get("center", [0.0, 0.0]), dtype=np.float32)
                size = np.asarray(region.get("size", [0.0, 0.0]), dtype=np.float32)
                d = np.abs(np.asarray(robot_xy, dtype=np.float32) - center) - size / 2.0
                best = min(best, float(np.linalg.norm(np.maximum(d, 0.0))))
        return 0.0 if not np.isfinite(best) else max(0.0, best)
