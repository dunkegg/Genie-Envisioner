
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

# from data.utils.domain_table import DomainTable
from data.utils.statistics import StatisticInfo
# from data.utils.get_actions import parse_h5

from utils import zero_rank_print
from data.utils.utils import intrinsic_transform, gen_crop_config, intrin_crop_transform

def load_jsonl(jsonl_path):
    """
    load jsonl file
    """
    data = []
    with open(jsonl_path, 'r', encoding='UTF-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


class CustomLeRobotDataset(Dataset):
    def __init__(self,
        data_roots,
        domains,
        task_recap_file = None,
        step_recap_file = None,
        sample_size=(192, 256), 
        sample_n_frames=64,
        preprocess = 'resize',
        valid_cam = ['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right'],
        chunk=1,
        action_chunk=None,
        n_previous=-1,
        previous_pick_mode='uniform',
        random_crop=True,
        dataset_info_cache_path = None,
        action_type = "absolute",
        action_space = "joint",
        ignore_seek = False,
        train_dataset=True,
        action_key = "action",
        state_key = "observation.state",
        bev_map_key = None,
        za_latent_key = None,
        za_latent_index_mode = "last",
        za_latent_tensor_key = None,
        beta_key = None,
        beta_index_mode = "last",
        use_unified_prompt = False,
        unified_prompt = "best quality, consistent and smooth motion, realistic, clear and distinct.",
        fix_epiidx = None,
        fix_sidx = None,
        fix_mem_idx = None,
        stat_file = None,
        stat_domain = None,
    
    ):
        """
        data_roots:              directory of LeRoBot dataset
        domains:                 name of your dataset, used to index different statistics
        task_recap_file:         json file of augmented task captions:
                                 {
                                    'ori_task_caption_1': ['new_caption_1', 'new_caption_2'...],
                                    'ori_task_caption_2': ['new_caption_1', 'new_caption_2'...],
                                 }
        step_recap_file:         json file of augmented step captions:
                                 {
                                    'ori_step_caption_1': ['new_caption_1', 'new_caption_2'...],
                                    'ori_step_caption_2': ['new_caption_1', 'new_caption_2'...],
                                 }
        sample_size:             video frame size
        sample_n_frames:         number of frames used to randomly or uniformly select memories
        preprocess:              frame preprocessing strategy, resize or center_crop_resize
        valid_cam:               list of cam names 
        chunk:                   number of video frames to predict
        action_chunk:            number of actions to predict, action_chunk should be an integer multiple of chunk.
        n_previous:              number of memory frames
        previous_pick_mode:      how to select memories
        random_crop:             randomly crop images
        dataset_info_cache_path: path to save dataset meta information cache
        action_type:             action space to use in this dataset
                                    'absolute': norm(act_t)
                                    'delta':    norm(act_t - act_{t-1})
                                    'relative': norm(act_t) - norm(state)
        action_space:            joint or eef, which is used to determinate the statistics values only in this dataset
        ignore_seek:             if True, load the first furture frame only
        use_unified_prompt:      if set all prompt the same
        unified_prompt:          unified prompt
        fix_epiidx:              used in validation stage only, set episode index to fix_epiidx
        fix_sidx:                used in validation stage only, set start index to fix_sidx
        fix_mem_idx:             used in validation stage only, set memory indexes to fix_mem_idx
        stat_file:               used to specific statistics
        stat_domain:             optional statistics domain name or domain-name map
        """
        
        zero_rank_print(f"loading annotations...")

        assert(action_type in ["delta", "absolute", "relative"])
        self.action_type = action_type
        assert(action_space in ["eef", "joint"])
        self.action_space = action_space



        self.action_key = action_key
        self.state_key = state_key
        self.bev_map_key = bev_map_key
        self.za_latent_key = za_latent_key
        self.za_latent_index_mode = za_latent_index_mode
        self.za_latent_tensor_key = za_latent_tensor_key
        self.beta_key = beta_key
        self.beta_index_mode = beta_index_mode

        self.random_crop = random_crop
        
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
            # construct the dataset_info
            for _data_root, _domain_name in zip(self.data_roots, domains):

                print(f"Loading {_domain_name} data from {_data_root}")
                
                # into the meta folder
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
                epiosdes_data = load_jsonl(episodes_jsonl) # episode_index  tasks  length

                for episode_data in tqdm(epiosdes_data):

                    episode_index = episode_data['episode_index']
                    tasks = episode_data['tasks']
                    if len(tasks) > 1:
                        task = random.choice(tasks)
                    else:
                        task = tasks[0]
                    length = episode_data['length']
                    
                    episode_chunk = int(episode_index//chunks_size)

                    parquet_path = os.path.join(data_folder, f"chunk-{episode_chunk:03d}", f"episode_{episode_index:06d}.parquet")
                    if not os.path.exists(parquet_path):
                        zero_rank_print(f"parquet file not found: {parquet_path}")
                        continue

                    video_path = os.path.join(video_folder, f"chunk-{episode_chunk:03d}", "{}", f"episode_{episode_index:06d}.mp4")
                    
                    info = [
                        video_path,
                        None, # no need for camera_info
                        parquet_path,
                        _domain_name, "", # DomainTable[_domain_name],
                        None, task, # no task_info
                        length,
                    ]
                    
                    self.dataset.append(info)

        if dataset_info_cache_path is not None and not(os.path.exists(dataset_info_cache_path)):
            zero_rank_print(f"Save Cache Dataset Information to {dataset_info_cache_path}")
            cache_dir = os.path.dirname(dataset_info_cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
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
                transforms.Resize(min(sample_size)),  # the size of shape (1,) means the smaller edge will be resized to it and the img will keep the h-w ratio.
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

        ### validation only
        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = fix_mem_idx

        ### load stat_file if provided
        self.StatisticInfo = StatisticInfo
        if stat_file is not None:
            with open(stat_file, "r") as f:
                self.StatisticInfo = json.load(f)
        self.stat_domain = stat_domain

        self.ignore_seek = ignore_seek

    def get_bev_map(self, data, parquet_path, vid_indexes):
        if self.bev_map_key is None:
            return None
        if self.bev_map_key not in data:
            raise KeyError(f"BEV map key '{self.bev_map_key}' not found in parquet data.")

        # Use exactly the same future-frame indexes as RGB (exclude memory frames).
        future_indexes = vid_indexes[self.n_previous:]
        bev_data = data[self.bev_map_key]
        maps = []
        for map_index in future_indexes:
            map_index = min(max(int(map_index), 0), len(bev_data) - 1)
            bev_map = bev_data.iloc[map_index] if hasattr(bev_data, "iloc") else bev_data[map_index]
            if isinstance(bev_map, str):
                candidates = [bev_map]
                if not os.path.isabs(bev_map):
                    dataset_root = os.path.dirname(os.path.dirname(os.path.dirname(parquet_path)))
                    candidates.insert(0, os.path.join(dataset_root, bev_map))
                bev_path = next((path for path in candidates if os.path.exists(path)), None)
                if bev_path is None:
                    raise FileNotFoundError(f"BEV map file not found for value: {bev_map}")
                bev_map = np.load(bev_path, allow_pickle=False)
            maps.append(torch.from_numpy(np.array(bev_map, dtype=np.float32, copy=True)))
        bev_maps = torch.stack(maps, dim=0)
        # Preserve the uint8 image convention through a reversible [-1,1] normalization.
        return bev_maps.div(127.5).sub(1.0).unsqueeze(1)

    def _load_latent_value(self, latent_value, parquet_path):
        if isinstance(latent_value, str):
            candidate_paths = [latent_value]
            if not os.path.isabs(latent_value):
                parquet_dir = os.path.dirname(parquet_path)
                data_dir = os.path.dirname(parquet_dir)
                domain_root = os.path.dirname(data_dir)
                candidate_paths = [
                    os.path.join(parquet_dir, latent_value),
                    os.path.join(data_dir, latent_value),
                    os.path.join(domain_root, latent_value),
                    latent_value,
                ]

            latent_path = None
            for candidate in candidate_paths:
                if os.path.exists(candidate):
                    latent_path = candidate
                    break
            if latent_path is None:
                raise FileNotFoundError(f"Za latent file not found for value: {latent_value}")

            try:
                latent_value = torch.load(latent_path, map_location="cpu", weights_only=True)
            except TypeError:
                latent_value = torch.load(latent_path, map_location="cpu")

        if isinstance(latent_value, dict):
            if self.za_latent_tensor_key is not None:
                latent_value = latent_value[self.za_latent_tensor_key]
            elif "latent" in latent_value:
                latent_value = latent_value["latent"]
            elif "za" in latent_value:
                latent_value = latent_value["za"]
            elif "z" in latent_value:
                latent_value = latent_value["z"]
            elif len(latent_value) == 1:
                latent_value = next(iter(latent_value.values()))
            else:
                raise KeyError(
                    "Za latent checkpoint is a dict. Set za_latent_tensor_key to choose the tensor."
                )

        return torch.as_tensor(latent_value, dtype=torch.float32).flatten()

    def get_za_latents(self, data, parquet_path, vid_indexes, action_indexes):
        if self.za_latent_key is None:
            return None
        if self.za_latent_key not in data:
            raise KeyError(f"Za latent key '{self.za_latent_key}' not found in parquet data.")

        if self.za_latent_index_mode == "last":
            latent_indexes = [int(action_indexes[-1])]
        elif self.za_latent_index_mode == "video":
            latent_indexes = [int(_) for _ in vid_indexes]
        elif self.za_latent_index_mode == "action":
            latent_indexes = [int(_) for _ in action_indexes]
        else:
            raise NotImplementedError(f"unsupported za_latent_index_mode: {self.za_latent_index_mode}")

        za_series = data[self.za_latent_key]
        za_latents = []
        for latent_index in latent_indexes:
            latent_index = min(max(latent_index, 0), len(za_series) - 1)
            latent_value = za_series.iloc[latent_index] if hasattr(za_series, "iloc") else za_series[latent_index]
            za_latents.append(self._load_latent_value(latent_value, parquet_path))

        za_latents = torch.stack(za_latents, dim=0)
        if self.za_latent_index_mode == "last":
            za_latents = za_latents[0]
        return za_latents

    def get_beta(self, data, vid_indexes, action_indexes):
        if self.beta_key is None:
            return None
        if self.beta_key not in data:
            raise KeyError(f"Beta key '{self.beta_key}' not found in parquet data.")

        if self.beta_index_mode == "last":
            beta_indexes = [int(action_indexes[-1])]
        elif self.beta_index_mode == "video":
            beta_indexes = [int(_) for _ in vid_indexes]
        elif self.beta_index_mode == "action":
            beta_indexes = [int(_) for _ in action_indexes]
        else:
            raise NotImplementedError(f"unsupported beta_index_mode: {self.beta_index_mode}")

        beta_series = data[self.beta_key]
        beta_values = []
        for beta_index in beta_indexes:
            beta_index = min(max(beta_index, 0), len(beta_series) - 1)
            beta_value = beta_series.iloc[beta_index] if hasattr(beta_series, "iloc") else beta_series[beta_index]
            beta_values.append(torch.from_numpy(np.array(beta_value, dtype=np.float32, copy=True)).flatten())

        beta_values = torch.stack(beta_values, dim=0)
        if self.beta_index_mode == "last":
            beta_values = beta_values[0]
        return beta_values

    def get_frame_indexes(self, total_frames, ):
        """
        select self.n_previous memory frames and self.action_chunk prediction frmaes
        1. randomly select the end frame
        2. take frames from {end-action_chunk} to {end} as the prediction frames
        3. uniformly/randomly select memory frames from {end-self.sample_n_frames} to {end-action_chunk}
        """

        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx+self.action_chunk))
            frame_indexes = action_indexes[::self.video_temporal_stride]
            action_indexes = np.clip(action_indexes, a_min=0, a_max=total_frames-1)
            frame_indexes = np.clip(frame_indexes, a_min=0, a_max=total_frames-1)
            # return self.fix_mem_idx + frame_indexes, self.fix_mem_idx + action_indexes #wzj fix
            fix_mem_idx = np.asarray(self.fix_mem_idx).reshape(-1)
            fix_mem_idx = np.clip(fix_mem_idx, a_min=0, a_max=total_frames - 1).astype(np.int64)

            # 如果 fix_mem_idx 是历史帧数组，则 future 从最后一个历史帧之后开始
            start_idx = int(fix_mem_idx[-1])

            vid_indexes = np.concatenate([fix_mem_idx, start_idx + frame_indexes])
            act_indexes = start_idx + action_indexes

            return vid_indexes, act_indexes

        chunk_end = random.randint(self.action_chunk, total_frames+self.action_chunk)

        indexes_start = max(-self.n_previous, chunk_end-self.sample_n_frames) ### prevent indexes including too many zeros  when sample_n_frames is much larger than the length of the episode
        indexes = np.array(list(range(indexes_start, chunk_end)))
        indexes = np.clip(indexes, a_min=1, a_max=total_frames-1).tolist()
        video_end = indexes[-self.action_chunk:]
        # mem_candidates = [
        #     indexes[int(i)] for i in range(0, self.sample_n_frames-self.action_chunk)
        # ]
        mem_candidates = indexes[:-self.action_chunk]
        if len(mem_candidates)<self.n_previous-1:
            mem_candidates = [1,]*(self.n_previous-1) + mem_candidates

        if self.previous_pick_mode == 'uniform':
            mem_indexes = [mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates)-1, self.n_previous).tolist()]

        elif self.previous_pick_mode == 'random':
            mem_indexes = [mem_candidates[i] for i in sorted(np.random.choice(list(range(0,len(mem_candidates)-1)), size=self.n_previous-1, replace=False).tolist())] + [mem_candidates[-1]]

        else:
            raise NotImplementedError(f"unsupported previous_pick_mode: {self.previous_pick_mode}")       

        if not self.ignore_seek:
            frame_indexes = mem_indexes + video_end[self.video_temporal_stride-1::self.video_temporal_stride]
        else:
            frame_indexes = mem_indexes + mem_indexes[-1:]

        action_indexes = mem_indexes + video_end

        return frame_indexes, action_indexes


    def get_action_bias_std(self, domain_name):
        stat_domain = domain_name
        if isinstance(self.stat_domain, dict):
            stat_domain = self.stat_domain.get(domain_name, domain_name)
        elif self.stat_domain is not None:
            stat_domain = self.stat_domain
            if domain_name.endswith("_state") and not stat_domain.endswith("_state"):
                stat_domain = stat_domain + "_state"
            elif domain_name.endswith("_delta") and not stat_domain.endswith("_delta"):
                stat_domain = stat_domain + "_delta"
        stat_key = stat_domain + "_" + self.action_space
        if stat_key not in self.StatisticInfo:
            raise KeyError(
                f"Statistics key '{stat_key}' not found. Available keys: {sorted(self.StatisticInfo.keys())}"
            )
        return torch.tensor(self.StatisticInfo[stat_key]['mean']).unsqueeze(0), torch.tensor(self.StatisticInfo[stat_key]['std']).unsqueeze(0)+1e-6


    def seek_mp4(self, video_path, cam_name_list, slices):
        """
        seek video frames according to the input slices;
        output video shape: (c,v,t,h,w)
        """
        video_list = []
        for cam_name in cam_name_list:
            video_reader = VideoFileClip(video_path.format(cam_name))
            fps = video_reader.fps
            video = []
            for idx in slices:
                video.append(video_reader.get_frame(float(idx)/fps))
            video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
            video = video.float()/255.
            video_reader.close()
            video_list.append(video)
        return video_list


    def transform_video(self, videos, specific_transforms_resize, intrinsics, sample_size):
        """
        crop (optional) and resize the videos, and modify the intrinsic accordingly
        """
        v = len(videos)
        new_videos = []
        new_intrinsics = []
        for iv in range(v):
            video = videos[iv]
            c, t, h, w = video.shape
            if self.random_crop:
                h_start, w_start, h_crop, w_crop = gen_crop_config(video)
                video = video[:,:,h_start:h_start+h_crop,w_start:w_start+w_crop]
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
        """
        input video should have shape (c,v,t,h,w)
        """
        c,v,t,h,w = video.shape
        video = specific_transforms_norm(video.permute(1,2,0,3,4).reshape(-1,c,h,w)).reshape(v,t,c,h,w).permute(2,0,1,3,4)
        return video


    def get_transform(self, ):
        sample_size = self.sample_size
        specific_transforms_resize = self.pixel_transforms_resize
        specific_transforms_norm = self.pixel_transforms_norm
        return sample_size, specific_transforms_resize, specific_transforms_norm


    def get_long_recaption(self, step_captions, task_caption):
        newcap = []
        # find = []
        for step_caption in step_captions:
            if self.step_recap_map is not None:
                recap_list = self.step_recap_map.get(step_caption,[])
                recap_list.append(step_caption)
                step_caption = np.random.choice(recap_list,1)
                newcap.append(str(step_caption[0]))
            else:
                newcap.append(step_caption)

        newcap = ", ".join(newcap)
        newcap = newcap.replace(" the "," ")
        if self.task_recap_map is not None:
            task_recap_list = self.task_recap_map.get(task_caption,[])
            task_recap_list.append(task_caption)
            task_newcap = np.random.choice(task_recap_list,1)
            task_newcap = str(task_newcap[0])
            fullcap = task_newcap + ": " + newcap
        else:
            task_newcap = task_caption
            fullcap = task_caption + ": " + newcap
        cap_type = random.randint(0,2)
        allcap = [fullcap, task_newcap, newcap]
        recap = allcap[cap_type]
        return recap



    def get_batch(self, idx):
        
        video_path = self.dataset[idx][0]
        parquet_path = self.dataset[idx][2]
        domain_name = self.dataset[idx][3]
        # domain_id = self.dataset[idx][4]
        caption = self.dataset[idx][6]
        total_frames = self.dataset[idx][7]
        
        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()
        vid_indexes, indexes = self.get_frame_indexes(total_frames, )
        
        data = pd.read_parquet(parquet_path)


        action_mean, action_std = self.get_action_bias_std(domain_name)
        state_mean, state_std = self.get_action_bias_std(domain_name + "_state")
        
        ###
        ### example data
        ### data[self.action_key] with the shape of T*C: [[1.0, 1.0, 1.0, ...], ...]
        ### data[self.state_key]  with the shape of T*C: [[1.0, 1.0, 1.0, ...], ...]
        try:
            action = np.stack([data[self.action_key][i] for i in range(data[self.action_key].shape[0])])
            state = np.stack([data[self.state_key][i] for i in range(data[self.state_key].shape[0])])
        except:
            raise ValueError("We currently only support action and state data with the shape of T*C!")
        
        action = action.astype(np.float32)
        state = state.astype(np.float32)
        state = torch.FloatTensor(state)[indexes][self.n_previous-1:self.n_previous]
        
        state = (state - state_mean) / state_std

        if self.action_type == "absolute":
            ### act = norm(act)

            action = action[indexes].astype(np.float32)
            action = torch.FloatTensor(action)
            action = (action - action_mean) / action_std

        elif self.action_type == "delta":
            ### delta_act = norm(act_{t} - act_{t-1})

            delta_act_meanv, delta_act_stdv = self.get_action_bias_std(domain_name + "_delta")
            action_curr = torch.FloatTensor(action[indexes].astype(np.float32))
            action_last = torch.FloatTensor(action[[_-1 for _ in indexes]].astype(np.float32))
            delta_action = action_curr - action_last
            ### keep current effector action
            delta_action[:, 6] = action_last[:, 6]
            delta_action[:, 13] = action_last[:, 13]
            delta_action = (delta_action - delta_act_meanv) / delta_act_stdv
            action = delta_action

        elif self.action_type == "relative":
            ### relative_act = norm(act) - norm(state)

            action_curr = action[indexes].astype(np.float32)
            action = torch.FloatTensor(action_curr)
            action = (action - action_mean) / action_std
            rel_action = action - state
            ### keep current effector action
            rel_action[:, 6] = action[:, 6]
            rel_action[:, 13] = action[:, 13]
            action = rel_action

        else:

            raise NotImplementedError


        videos = self.seek_mp4(video_path, self.valid_cam, vid_indexes)

        videos, _ = self.transform_video(
            videos, specific_transforms_resize, None, sample_size
        )
        videos = self.normalize_video(videos, specific_transforms_norm)

        bev_map = self.get_bev_map(data, parquet_path, vid_indexes)
        za_latents = self.get_za_latents(data, parquet_path, vid_indexes, indexes)
        beta = self.get_beta(data, vid_indexes, indexes)

        return videos, action, caption, state, bev_map, za_latents, beta


    def __len__(self):
        return self.length


    def __getitem__(self, idx):        
        
        # video, actions, caption, state = self.get_batch(idx)

        if self.fix_epiidx is not None:
            video, actions, caption, state, bev_map, za_latents, beta = self.get_batch(self.fix_epiidx)
        else:
            while True:
                try:
                    video, actions, caption, state, bev_map, za_latents, beta = self.get_batch(idx)
                    break
                except:
                    ### print error information to debug
                    traceback.print_exc()
                    ### 
                    idx = random.randint(0, self.length-1)
                    
        sample = dict(
            video=video,
            actions=actions,
            caption=caption,
            state=state,
        )
        if bev_map is not None:
            sample[self.bev_map_key] = bev_map
        if za_latents is not None:
            sample["za_latents"] = za_latents
        if beta is not None:
            sample["beta"] = beta
        return sample
