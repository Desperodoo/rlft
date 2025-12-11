import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from h5py import Dataset, File, Group
from torch.utils.data.sampler import Sampler


class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.
    Resampling from it until a specified number of iterations have been sampled
    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


TARGET_KEY_TO_SOURCE_KEY = {
    "states": "env_states",
    "observations": "obs",
    "success": "success",
    "next_observations": "obs",
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    "actions": "actions",
}


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def load_traj_hdf5(path, num_traj=None):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
        keys = keys[:num_traj]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    print("Loaded")
    return ret


def load_demo_dataset(
    path, keys=["observations", "actions"], num_traj=None, concat=True
):
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = raw_data["traj_0"]
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in raw_data]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[:-1] for t in dataset[target_key]], axis=0
                )
            elif target_key in ["next_observations", "next_states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[1:] for t in dataset[target_key]], axis=0
                )
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print("Load", target_key, dataset[target_key].shape)
        else:
            print(
                "Load",
                target_key,
                len(dataset[target_key]),
                type(dataset[target_key][0]),
            )
    return dataset


def convert_obs(obs, concat_fn, transpose_fn, state_obs_extractor, depth = True):
    img_dict = obs["sensor_data"]
    ls = ["rgb"]
    if depth:
        ls = ["rgb", "depth"]

    new_img_dict = {
        key: transpose_fn(
            concat_fn([v[key] for v in img_dict.values()])
        )  # (C, H, W) or (B, C, H, W)
        for key in ls
    }
    if "depth" in new_img_dict and isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
        new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

    # Unified version
    states_to_stack = state_obs_extractor(obs)
    for j in range(len(states_to_stack)):
        if states_to_stack[j].dtype == np.float64:
            states_to_stack[j] = states_to_stack[j].astype(np.float32)
    try:
        state = np.hstack(states_to_stack)
    except:  # dirty fix for concat trajectory of states
        state = np.column_stack(states_to_stack)
    if state.dtype == np.float64:
        for x in states_to_stack:
            print(x.shape, x.dtype)
        import pdb

        pdb.set_trace()

    out_dict = {
        "state": state,
        "rgb": new_img_dict["rgb"],
    }

    if "depth" in new_img_dict:
        out_dict["depth"] = new_img_dict["depth"]


    return out_dict


def build_obs_space(env, depth_dtype, state_obs_extractor):
    # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env
    obs_space = env.observation_space

    # Unified version
    state_dim = sum([v.shape[0] for v in state_obs_extractor(obs_space)])

    single_img_space = next(iter(env.observation_space["image"].values()))
    h, w, _ = single_img_space["rgb"].shape
    n_images = len(env.observation_space["image"])

    return spaces.Dict(
        {
            "state": spaces.Box(
                -float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32
            ),
            "rgb": spaces.Box(0, 255, shape=(n_images * 3, h, w), dtype=np.uint8),
            "depth": spaces.Box(
                -float("inf"), float("inf"), shape=(n_images, h, w), dtype=depth_dtype
            ),
        }
    )


def build_state_obs_extractor(env_id):
    # NOTE: You can tune/modify state observations specific to each environment here as you wish. By default we include all data
    # but in some use cases you might want to exclude e.g. obs["agent"]["qvel"] as qvel is not always something you query in the real world.
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())


class AgentWrapper(nn.Module):
    """Wrapper that combines visual encoder and agent for evaluation.
    
    Aligns with train_rgbd.py's Agent.encode_obs and Agent.get_action methods.
    Can be used by both offline (train_offline_rl.py) and online (train_online_finetune.py) training.
    
    Args:
        agent: The policy agent (diffusion policy, flow matching, etc.)
        visual_encoder: Visual encoder for RGB/depth images (can be None for state-only)
        include_rgb: Whether to include RGB in observations
        include_depth: Whether to include depth in observations
        obs_horizon: Number of observation frames to stack
        act_horizon: Number of action frames to predict (optional, for slicing output)
    """
    def __init__(self, agent, visual_encoder, include_rgb, include_depth, obs_horizon, act_horizon=None):
        super().__init__()
        self.agent = agent
        self.visual_encoder = visual_encoder
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon if act_horizon is not None else obs_horizon
        
    def encode_obs(self, obs_seq, eval_mode=True):
        """Encode observations to get obs_cond for agents.
        
        Mirrors train_rgbd.py Agent.encode_obs exactly.
        
        Input from environment:
            obs_seq["rgb"]: (B, obs_horizon, H, W, C) uint8
            obs_seq["state"]: (B, obs_horizon, state_dim) float32
        
        Output:
            obs_cond: (B, obs_horizon * (visual_dim + state_dim))
        """
        # Convert from NHWC to NCHW if needed (environment returns HWC)
        img_seq = None
        
        if self.include_rgb and "rgb" in obs_seq:
            rgb = obs_seq["rgb"]
            if rgb.dim() == 5 and rgb.shape[-1] in [1, 3, 4]:
                # (B, T, H, W, C) -> (B, T, C, H, W)
                rgb = rgb.permute(0, 1, 4, 2, 3)
            rgb = rgb.float() / 255.0
            img_seq = rgb
            
        if self.include_depth and "depth" in obs_seq:
            depth = obs_seq["depth"]
            if depth.dim() == 5 and depth.shape[-1] == 1:
                depth = depth.permute(0, 1, 4, 2, 3)
            depth = depth.float() / 1024.0
            img_seq = depth
            
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W)
        
        features_list = []
        
        if self.visual_encoder is not None and img_seq is not None:
            batch_size = img_seq.shape[0]
            img_seq_flat = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
            visual_feature = self.visual_encoder(img_seq_flat)  # (B*obs_horizon, D)
            visual_feature = visual_feature.reshape(
                batch_size, self.obs_horizon, visual_feature.shape[1]
            )  # (B, obs_horizon, D)
            features_list.append(visual_feature)
        
        # State
        state = obs_seq["state"]  # (B, obs_horizon, state_dim)
        features_list.append(state)
        
        # Concatenate: (B, obs_horizon, D+state_dim)
        feature = torch.cat(features_list, dim=-1)
        # Flatten: (B, obs_horizon * (D+state_dim))
        obs_cond = feature.flatten(start_dim=1)
        
        return obs_cond
    
    def get_action(self, obs_seq, **kwargs):
        """Get action from observations (for evaluation).
        
        Mirrors train_rgbd.py Agent.get_action.
        
        Args:
            obs_seq: Dict with 'rgb', 'depth', 'state' observations
            **kwargs: Additional arguments to pass to agent.get_action()
                - For ReinFlowAgent: deterministic=True, use_ema=True
        """
        with torch.no_grad():
            # encode_obs returns (B, global_cond_dim) - flattened observation features
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            
            # Pass flattened obs_cond directly to agent.get_action
            # The agent's velocity_net/noise_pred_net expects global_cond as (B, global_cond_dim)
            result = self.agent.get_action(obs_cond, **kwargs)
            
            # Handle agents that return (action_seq, chains) tuple (e.g., ReinFlowAgent)
            if isinstance(result, tuple):
                action_seq = result[0]
            else:
                action_seq = result
            
            # Only return act_horizon actions (aligned with train_rgbd.py)
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            return action_seq[:, start:end]
    
    def eval(self):
        self.agent.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.eval()
        return self
    
    def train(self):
        self.agent.train()
        if self.visual_encoder is not None:
            self.visual_encoder.train()
        return self
