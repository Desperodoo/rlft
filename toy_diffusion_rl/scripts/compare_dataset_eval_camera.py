#!/usr/bin/env python3
"""
Compare Camera Views: Dataset vs Evaluation Environment

This script compares the image views from:
1. The dataset (peg_insertion_side_cpu.h5) - saves first episode as MP4
2. The evaluation environment - runs a random episode and saves as MP4

This helps diagnose any camera view mismatch between training data and evaluation.

Usage:
    python scripts/compare_dataset_eval_camera.py \
        --dataset_path data/peg_insertion_side_cpu.h5 \
        --output_dir results/camera_comparison \
        --task PegInsertionSide-v1
"""

import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)

import argparse
import os
import sys
import h5py
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
    MANISKILL_AVAILABLE = True
except ImportError:
    MANISKILL_AVAILABLE = False
    print("Warning: ManiSkill3 not available")


def save_images_as_mp4(images: List[np.ndarray], output_path: str, fps: int = 30):
    """Save a list of images as an MP4 video.
    
    Args:
        images: List of RGB images (H, W, 3) uint8
        output_path: Path to save the MP4 file
        fps: Frames per second
    """
    if len(images) == 0:
        print(f"No images to save for {output_path}")
        return
    
    # Get image dimensions
    h, w = images[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img in images:
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    
    out.release()
    print(f"Saved video: {output_path} ({len(images)} frames, {h}x{w})")


def extract_dataset_images(dataset_path: str, episode_id: int = 0) -> List[np.ndarray]:
    """Extract images from a specific episode in the dataset.
    
    Args:
        dataset_path: Path to the HDF5 dataset
        episode_id: Episode ID to extract
    
    Returns:
        List of RGB images
    """
    with h5py.File(dataset_path, 'r') as f:
        episode_ids = f['episode_ids'][:]
        images_data = f['images']
        
        # Find indices for the specified episode
        episode_mask = episode_ids == episode_id
        episode_indices = np.where(episode_mask)[0]
        
        if len(episode_indices) == 0:
            print(f"Episode {episode_id} not found in dataset")
            return []
        
        print(f"Episode {episode_id}: {len(episode_indices)} frames")
        
        # Extract images for this episode
        images = []
        for idx in episode_indices:
            img = images_data[idx]
            images.append(img)
        
        return images


def extract_eval_images_all_cameras(
    task: str = "PegInsertionSide-v1",
    num_steps: int = 200,
    seed: int = 42,
    image_size: int = 128,
) -> Dict[str, List[np.ndarray]]:
    """Run evaluation and extract images from ALL cameras.
    
    Args:
        task: ManiSkill task name
        num_steps: Number of steps to run
        seed: Random seed
        image_size: Target image size
    
    Returns:
        Dictionary mapping camera names to lists of RGB images
    """
    if not MANISKILL_AVAILABLE:
        raise ImportError("ManiSkill3 not available")
    
    # Create environment with raw sensor data access
    env = gym.make(
        task,
        obs_mode="state_dict+rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
    )
    env = CPUGymWrapper(env)
    
    # Collect images from all cameras
    camera_images: Dict[str, List[np.ndarray]] = {}
    
    obs, info = env.reset(seed=seed)
    
    # Get camera names
    if "sensor_data" in obs:
        camera_names = list(obs["sensor_data"].keys())
        print(f"Found cameras: {camera_names}")
        for cam_name in camera_names:
            camera_images[cam_name] = []
    else:
        print("No sensor_data found in observation")
        env.close()
        return {}
    
    # Also collect the "first camera" that the current code uses
    camera_images["first_camera_in_loop"] = []
    
    for step in range(num_steps):
        # Extract images from all cameras
        for cam_name in camera_names:
            if "rgb" in obs["sensor_data"][cam_name]:
                rgb = obs["sensor_data"][cam_name]["rgb"]
                if hasattr(rgb, 'cpu'):
                    rgb = rgb.cpu().numpy()
                if len(rgb.shape) == 4:
                    rgb = rgb[0]  # Unbatch
                if rgb.dtype != np.uint8:
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
                    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
                camera_images[cam_name].append(rgb.copy())
        
        # Also record which camera the current loop-based extraction uses
        for cam_name, cam_data in obs["sensor_data"].items():
            if "rgb" in cam_data:
                rgb = cam_data["rgb"]
                if hasattr(rgb, 'cpu'):
                    rgb = rgb.cpu().numpy()
                if len(rgb.shape) == 4:
                    rgb = rgb[0]
                if rgb.dtype != np.uint8:
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
                    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
                camera_images["first_camera_in_loop"].append(rgb.copy())
                break  # This is what the current code does - take first camera
        
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    
    return camera_images


def compare_images_visually(
    dataset_images: List[np.ndarray],
    eval_images_dict: Dict[str, List[np.ndarray]],
    output_dir: str,
):
    """Create side-by-side comparison videos.
    
    Args:
        dataset_images: Images from dataset
        eval_images_dict: Dictionary of camera name -> images from evaluation
        output_dir: Directory to save comparison videos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine minimum length for comparison
    min_len = len(dataset_images)
    
    for cam_name, eval_images in eval_images_dict.items():
        if len(eval_images) == 0:
            continue
            
        comparison_len = min(min_len, len(eval_images))
        
        # Create side-by-side comparison frames
        comparison_frames = []
        for i in range(comparison_len):
            dataset_img = dataset_images[i]
            eval_img = eval_images[i]
            
            # Ensure same size
            h, w = dataset_img.shape[:2]
            if eval_img.shape[:2] != (h, w):
                eval_img = cv2.resize(eval_img, (w, h), interpolation=cv2.INTER_AREA)
            
            # Add labels
            dataset_labeled = dataset_img.copy()
            eval_labeled = eval_img.copy()
            
            # Put text labels (convert to BGR for cv2.putText, then back to RGB)
            cv2.putText(dataset_labeled, "Dataset", (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(eval_labeled, f"Eval:{cam_name}", (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Concatenate side by side
            combined = np.concatenate([dataset_labeled, eval_labeled], axis=1)
            comparison_frames.append(combined)
        
        # Save comparison video
        safe_cam_name = cam_name.replace("/", "_").replace(" ", "_")
        comparison_path = os.path.join(output_dir, f"comparison_dataset_vs_{safe_cam_name}.mp4")
        save_images_as_mp4(comparison_frames, comparison_path, fps=30)


def analyze_camera_differences(
    dataset_images: List[np.ndarray],
    eval_images_dict: Dict[str, List[np.ndarray]],
):
    """Compute image similarity metrics between dataset and eval cameras.
    
    This helps identify which camera from eval best matches the dataset.
    """
    print("\n" + "=" * 60)
    print("Camera Similarity Analysis")
    print("=" * 60)
    
    # Use first N frames for comparison
    num_compare = min(50, len(dataset_images))
    dataset_sample = dataset_images[:num_compare]
    
    for cam_name, eval_images in eval_images_dict.items():
        if len(eval_images) < num_compare:
            print(f"{cam_name}: Not enough frames for comparison")
            continue
        
        eval_sample = eval_images[:num_compare]
        
        # Compute mean absolute difference
        diffs = []
        for i in range(num_compare):
            d_img = dataset_sample[i].astype(np.float32)
            e_img = eval_sample[i].astype(np.float32)
            
            # Resize if needed
            if d_img.shape != e_img.shape:
                e_img = cv2.resize(eval_sample[i], 
                                   (d_img.shape[1], d_img.shape[0]),
                                   interpolation=cv2.INTER_AREA).astype(np.float32)
            
            diff = np.abs(d_img - e_img).mean()
            diffs.append(diff)
        
        mean_diff = np.mean(diffs)
        print(f"{cam_name:30s}: Mean pixel diff = {mean_diff:.2f} (lower is more similar)")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare camera views between dataset and evaluation environment"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/peg_insertion_side_cpu.h5",
        help="Path to the HDF5 dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/camera_comparison",
        help="Directory to save output videos"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="PegInsertionSide-v1",
        help="ManiSkill task name"
    )
    parser.add_argument(
        "--episode_id",
        type=int,
        default=0,
        help="Episode ID to extract from dataset"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Number of steps to run in evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Target image size"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Camera View Comparison: Dataset vs Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Task: {args.task}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Step 1: Extract images from dataset
    print("Step 1: Extracting images from dataset...")
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.dataset_path
    ) if not os.path.isabs(args.dataset_path) else args.dataset_path
    
    dataset_images = extract_dataset_images(dataset_path, args.episode_id)
    
    if len(dataset_images) > 0:
        dataset_video_path = os.path.join(args.output_dir, "dataset_episode.mp4")
        save_images_as_mp4(dataset_images, dataset_video_path, fps=30)
    
    # Step 2: Extract images from evaluation (all cameras)
    print("\nStep 2: Extracting images from evaluation environment...")
    eval_images_dict = extract_eval_images_all_cameras(
        task=args.task,
        num_steps=args.eval_steps,
        seed=args.seed,
        image_size=args.image_size,
    )
    
    # Save individual camera videos
    for cam_name, images in eval_images_dict.items():
        if len(images) > 0:
            safe_cam_name = cam_name.replace("/", "_").replace(" ", "_")
            video_path = os.path.join(args.output_dir, f"eval_{safe_cam_name}.mp4")
            save_images_as_mp4(images, video_path, fps=30)
    
    # Step 3: Create comparison videos
    print("\nStep 3: Creating comparison videos...")
    compare_images_visually(dataset_images, eval_images_dict, args.output_dir)
    
    # Step 4: Analyze camera differences
    print("\nStep 4: Analyzing camera differences...")
    analyze_camera_differences(dataset_images, eval_images_dict)
    
    print("\nDone! Check the output directory for videos:")
    print(f"  {args.output_dir}/")


if __name__ == "__main__":
    main()
