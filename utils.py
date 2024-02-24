"""The functions for supporting the MCL experiments
"""

import json
import os.path

import numpy as np
from scipy.spatial.transform import Rotation as R

from evo.core import metrics, sync
from evo.tools import file_interface

from nof.dataset.ray_utils import get_ray_directions


def load_data(pose_path, max_beams=None):
    # Read and parse the poses
    timestamps = []
    poses_gt = []
    odoms = []
    scans = []
    params = {}

    try:
        with open(pose_path, 'r') as f:
            all_data = json.load(f)

        # downsample beams number according to the max_beams
        if max_beams is not None:
            downsample_factor = all_data['num_beams'] // max_beams
            all_data['angle_res'] *= downsample_factor
            all_data['num_beams'] = max_beams
            all_data['angle_max'] = all_data['angle_min'] + all_data['angle_res'] * max_beams

        params.update({'num_beams': all_data['num_beams']})
        params.update({'angle_min': all_data['angle_min']})
        params.update({'angle_max': all_data['angle_max']})
        params.update({'angle_res': all_data['angle_res']})
        params.update({'max_range': all_data['max_range']})

        near = 0.02
        far = np.floor(all_data['max_range'])
        bound = np.array([near, far])
        # ray directions for all beams in the lidar coordinate, shape: (N, 2)
        directions = get_ray_directions(all_data['angle_min'], all_data['angle_max'],
                                        all_data['angle_res'])

        params.update({'near': near})
        params.update({'far': far})
        params.update({'bound': bound})
        params.update({'directions': directions})

        for data in all_data['scans']:
            timestamps.append(data['timestamp'])

            pose = data['pose_gt']
            poses_gt.append(pose)

            odom = data['odom_reading']
            odoms.append(odom)

            scan = np.array(data['range_reading'])
            if max_beams is not None:
                scan = scan[::downsample_factor][:max_beams]
            scan[scan >= all_data['max_range']] = 0
            scans.append(scan)

    except FileNotFoundError:
        print('Ground truth poses are not available.')

    return np.array(timestamps), np.array(poses_gt), np.array(odoms), np.array(scans), params


def particles2pose(particles):
    """
    Convert particles to the estimated pose accodring to the particles' distribution
    :param particles: 2-D array, (N, 4) shape
    :return: a estimated 2D pose, 1-D array, (3,) shape
    """
    normalized_weight = particles[:, 3] / np.sum(particles[:, 3])

    # average angle (https://vicrucann.github.io/tutorials/phase-average/)
    particles_mat = np.zeros_like(particles)
    particles_mat[:, :2] = particles[:, :2]
    particles_mat[:, 2] = np.cos(particles[:, 2])
    particles_mat[:, 3] = np.sin(particles[:, 2])
    estimated_pose_temp = particles_mat.T.dot(normalized_weight.T)

    estimated_pose = np.zeros(shape=(3,))
    estimated_pose[:2] = estimated_pose_temp[:2]
    estimated_pose[2] = np.arctan2(estimated_pose_temp[-1], estimated_pose_temp[-2])

    return estimated_pose


def get_est_poses(all_particles, start_idx, numParticles):
    estimated_traj = []
    ratio = 0.8

    for frame_idx in range(start_idx, all_particles.shape[0]):
        particles = all_particles[frame_idx]
        # collect top 80% of particles to estimate pose
        idxes = np.argsort(particles[:, 3])[::-1]
        idxes = idxes[:int(ratio * numParticles)]

        partial_particles = particles[idxes]
        if np.sum(partial_particles[:, 3]) == 0:
            continue

        estimated_pose = particles2pose(partial_particles)
        estimated_traj.append(estimated_pose)

    estimated_traj = np.array(estimated_traj)

    return estimated_traj


def convert2tum(timestamps, poses):
    tum_poses = []

    for t, pose in zip(timestamps, poses):
        x, y, yaw = pose
        q = R.from_euler('z', yaw).as_quat()
        curr_data = [t,
                     x, y, 0,
                     q[0], q[1], q[2], q[3]]

        tum_poses.append(curr_data)

    tum_poses = np.array(tum_poses)

    return tum_poses


def evaluate_APE(est_poses, gt_poses, use_converge=False):
    # align est and gt
    max_diff = 0.01
    traj_ref, traj_est = sync.associate_trajectories(gt_poses, est_poses, max_diff)
    data = (traj_ref, traj_est)

    # location error
    ape_location = metrics.APE(metrics.PoseRelation.translation_part)
    ape_location.process_data(data)
    location_errors = ape_location.error

    location_ptc5 = location_errors < 0.05
    location_ptc5 = np.sum(location_ptc5) / location_ptc5.shape[0] * 100

    location_ptc10 = location_errors < 0.1
    location_ptc10 = np.sum(location_ptc10) / location_ptc10.shape[0] * 100

    location_ptc20 = location_errors < 0.2
    location_ptc20 = np.sum(location_ptc20) / location_ptc20.shape[0] * 100

    location_rmse = ape_location.get_statistic(metrics.StatisticsType.rmse) * 100

    # yaw error
    ape_yaw = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_yaw.process_data(data)

    yaw_errors = ape_yaw.error
    yaw_ptc5 = yaw_errors < 0.5
    yaw_ptc5 = np.sum(yaw_ptc5) / yaw_ptc5.shape[0] * 100

    yaw_ptc10 = yaw_errors < 1.0
    yaw_ptc10 = np.sum(yaw_ptc10) / yaw_ptc10.shape[0] * 100

    yaw_ptc20 = yaw_errors < 2.0
    yaw_ptc20 = np.sum(yaw_ptc20) / yaw_ptc20.shape[0] * 100

    yaw_rmse = ape_yaw.get_statistic(metrics.StatisticsType.rmse)

    if use_converge:
        converge_idx = 0
        for idx in range(location_errors.shape[0]):
            if location_errors[idx] < 0.5 and yaw_errors[idx] < 10:
                converge_idx = idx
                break
        location_rmse = np.sqrt(np.mean(location_errors[converge_idx:] ** 2)) * 100
        yaw_rmse = np.sqrt(np.mean(yaw_errors[converge_idx:] ** 2))

    return location_rmse, location_ptc5, location_ptc10, location_ptc20, \
           yaw_rmse, yaw_ptc5, yaw_ptc10, yaw_ptc20


def summary_loc(loc_results, start_idx, numParticles, timestamps,
                result_dir, gt_file, init_time_thres=20, use_converge=False):
    # convert loc_results to tum format
    timestamps = timestamps[start_idx:]

    # get estimated poses
    est_poses = get_est_poses(loc_results, start_idx, numParticles)
    est_tum = convert2tum(timestamps, est_poses)

    # save est_traj in tum format
    est_tum_file = os.path.join(result_dir, 'IRMCL.txt')
    np.savetxt(est_tum_file, est_tum)

    # evo evaluation
    print('\nEvaluation')

    # Estimated poses
    est_poses = file_interface.read_tum_trajectory_file(est_tum_file)
    est_poses.reduce_to_time_range(init_time_thres)

    # GT
    gt_poses = file_interface.read_tum_trajectory_file(gt_file)
    gt_poses.reduce_to_time_range(init_time_thres)

    print("Sequence information: ", gt_poses)
    print(("{:>15}\t" * 8).format(
        "location_rmse", "location_ptc5", "location_ptc10", "location_ptc20",
        "yaw_rmse", "yaw_ptc5", "yaw_ptc10", "yaw_ptc20"))

    location_rmse, location_ptc5, location_ptc10, location_ptc20, \
    yaw_rmse, yaw_ptc5, yaw_ptc10, yaw_ptc20 = \
        evaluate_APE(est_poses, gt_poses, use_converge=use_converge)

    # print error info
    print(("{:15.2f}\t" * 8).format(
        location_rmse, location_ptc5, location_ptc10, location_ptc20,
        yaw_rmse, yaw_ptc5, yaw_ptc10, yaw_ptc20))


if __name__ == '__main__':
    demo_results = np.load('./results/ipblab/loc_test/test1/loc_results.npz')
    # loading localization results
    timestamps = demo_results['timestamps']
    particles = demo_results['particles']
    start_idx = demo_results['start_idx']
    numParticles = demo_results['numParticles']

    gt_file ='./data/ipblab/loc_test/test1/seq_1_gt_pose.txt'
    result_dir = './results/ipblab/loc_test/test1/'

    summary_loc(particles, start_idx, numParticles, timestamps, result_dir, gt_file)
