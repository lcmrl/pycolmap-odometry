import os
import cv2
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from vismatch import get_matcher
from vismatch.viz import plot_matches

from pyquaternion import Quaternion
from src.odometry.odometry import VisualOdometry

import cProfile
import pstats

REINIZIALIZE_AFTER = -1  # Reinitialize after this many frames, -1 disables

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", type=Path, required=True)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-a", "--camera", type=Path, required=True)
    parser.add_argument("-r", "--rig", type=Path, required=True)
    parser.add_argument("-w", "--work_dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.camera) as f:
        camera_config = yaml.safe_load(f)
    with open(args.rig) as f:
        rig_config = yaml.safe_load(f)

    start_frame, end_frame = config['general']['frames_range']
    working_dir = args.work_dir
    frames_dir = args.images
    cam0_dir = frames_dir / "cam0"

    # Use sorted Path.glob for better performance
    frames_cam0 = sorted([f.name for f in cam0_dir.iterdir() if f.is_file()])
    if end_frame == -1:
        end_frame = len(frames_cam0)

    # Prepare output files
    out_file_path = working_dir / "trajectory.txt"
    out_images_file_path = working_dir / "images.txt"
    for path in [out_file_path, out_images_file_path]:
        if path.exists():
            path.unlink()

    visual_odometry = VisualOdometry(
        working_dir=working_dir,
        config=config,
        camera_config=camera_config,
        rig_config=rig_config,
    )

    pose_changes = []

    images_t0 = {}
    keypoints_t0 = {}
    descriptors_t0 = {}

    images_t1 = {}
    keypoints_t1 = {}
    descriptors_t1 = {}

    matches = {}

    matcher = get_matcher("superpoint-lightglue", device="cuda")
    #matcher.skip_ransac = True  # Disable RANSAC in matcher, we'll handle it in VisualOdometry
    img_size = 512

    # Precompute reinitialization frame set for O(1) check
    reinit_set = {REINIZIALIZE_AFTER} if REINIZIALIZE_AFTER >= 0 else set()

    if config['general']['save_keyframes']:
        keyframes_log = open(working_dir / "keyframes_log.txt", "w")

    for frame_counter, frame_index in enumerate(tqdm(range(start_frame, end_frame))):
        try:
            img_name = frames_cam0[frame_index]
        except:
            continue
        images = []

        for cam in visual_odometry.cameras:
            img_path = frames_dir / cam / img_name
            images_t1[cam] = img_path
        
        if frame_counter == 0:
            images_t0 = images_t1.copy()
            keypoints_t0 = keypoints_t1.copy()
            descriptors_t0 = descriptors_t1.copy()
            continue
        
        # Keyframe selection on optical flow
        img_cam0_t0 = matcher.load_image(images_t0["cam0"], resize=img_size)
        img_cam0_t1 = matcher.load_image(images_t1["cam0"], resize=img_size)
        mathes_t0_t1 = matcher(img_cam0_t0, img_cam0_t1)
        plot_matches(img_cam0_t0, img_cam0_t1, mathes_t0_t1, save_path="./assets/output/img.png")

        mpts_t0 = torch.tensor(mathes_t0_t1['matched_kpts0'], dtype=torch.float32)
        mpts_t1 = torch.tensor(mathes_t0_t1['matched_kpts1'], dtype=torch.float32)
        match_dist = torch.linalg.norm(mpts_t0 - mpts_t1, dim=1)
        median_match_dist = torch.median(match_dist).item()

        if median_match_dist > config['mapping']['max_match_distance']:
            print(f"Matching {images_t0['cam0']} <-> {images_t1['cam0']} {median_match_dist}")
            images_t0 = images_t1.copy()
            keypoints_t0 = keypoints_t1.copy()
            descriptors_t0 = descriptors_t1.copy()

        print(median_match_dist)
        continue

        reinitialize = frame_index in reinit_set
        pose_change, log = visual_odometry.run(img_name, images, reinitialize=reinitialize);print(pose_change)
        cov = log['current_frame']['covariance_matrix_angles_translations']

        if cov is None:
            cov = 999
        else:
            cov = np.diag(cov)[-3:]
            cov = np.linalg.norm(np.sqrt(cov))

        pose_changes.append((pose_change, cov))

        if config['general']['log']:
            #print(log)
            print(f"\033[91mstatus: {log['current_frame']['status']}\033[0m" if log['current_frame']['status'] is False else f"\033[92mstatus: {log['current_frame']['status']}\033[0m")
            print(f"is_keyframe: {log['current_frame']['is_keyframe']}")
            print(f"num_features: {log['current_frame']['num_features']}")
            print(f"corrupted_master_image: {log['current_frame']['corrupted_master_image']}")
            print(f"corrupted_slave_image: {log['current_frame']['corrupted_slave_image']}")
            print(f"not_enough_features_on_master: {log['current_frame']['not_enough_features_on_master']}")
            print(f"inlier_matches: {log['current_frame']['inlier_matches']}")
            print(f"inlier_ratio: {log['current_frame']['inlier_ratio']}")
            print(f"covariance_matrix_angles_translations: {log['current_frame']['covariance_matrix_angles_translations']}")

        if log['current_frame']['is_keyframe'] and config['general']['save_keyframes']:
            keyframes_log.write(f"{log['current_frame']['frame_name']}\n")

    keyframes_log.close() if config['general']['save_keyframes'] else None
    summary = visual_odometry.get_performance_summary()
    print(summary)

    # Write output in bulk to improve speed
    traj_lines = []
    img_lines = []
    t_cumulative = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    q_cumulative = Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))

    for changes, cov in pose_changes:
        try:
            #image, id, delta_t, delta_q, t_cumulative, q_cumulative = changes[0]
            image, id, delta_t, delta_q, _, _ = changes[0]
            #t_ = -q_cumulative.rotation_matrix @ t_cumulative
            #traj_lines.append(f"{image} {t_cumulative[0]} {t_cumulative[1]} {t_cumulative[2]} {cov}\n")
            #img_lines.append(f"{id} {q_cumulative[0]} {q_cumulative[1]} {q_cumulative[2]} {q_cumulative[3]} "
            #                 f"{t_[0]} {t_[1]} {t_[2]} 1 {image}\n\n")
            
            t_cumulative = t_cumulative + q_cumulative.inverse.rotate(delta_t)
            q_cumulative = delta_q * q_cumulative
            t_ = -q_cumulative.rotation_matrix @ t_cumulative
            traj_lines.append(f"{image} {t_cumulative[0]} {t_cumulative[1]} {t_cumulative[2]} {cov}\n")
            img_lines.append(f"{id} {q_cumulative[0]} {q_cumulative[1]} {q_cumulative[2]} {q_cumulative[3]} "
                             f"{t_[0]} {t_[1]} {t_[2]} 1 {image}\n\n")
            
        except Exception:
            continue

    out_file_path.write_text("".join(traj_lines))
    out_images_file_path.write_text("".join(img_lines))

if __name__ == "__main__":

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    
    # Print the top 20 most time-consuming functions
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)