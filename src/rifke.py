import torch

from einops import rearrange
import numpy as np

from torch import Tensor

from .geometry import axis_angle_rotation, matrix_to_axis_angle
from .joints import INFOS


def joints_to_rifke(joints, jointstype="smpljoints"):
    # Joints to rotation invariant poses (Holden et. al.)
    # Similar function than fke2rifke in Language2Pose repository
    # Adapted from the pytorch version of TEMOS
    # https://github.com/Mathux/TEMOS
    # Estimate the last velocities based on acceleration
    # Difference of rotations are in SO3 space now

    # First remove the ground
    ground = joints[..., 2].min()
    poses = joints.clone()
    poses[..., 2] -= ground

    poses = joints.clone()
    translation = poses[..., 0, :].clone()

    # Let the root have the Z translation --> gravity axis
    root_grav_axis = translation[..., 2]

    # Trajectory => Translation without gravity axis (Z)
    trajectory = translation[..., [0, 1]]

    # Compute the forward direction (before removing the root joint)
    forward = get_forward_direction(poses, jointstype=jointstype)

    # Delete the root joints of the poses
    poses = poses[..., 1:, :]

    # Remove the trajectory of the poses
    poses[..., [0, 1]] -= trajectory[..., None, :]

    vel_trajectory = torch.diff(trajectory, dim=-2)

    # repeat the last acceleration
    # for the last (not seen) velocity
    last_acceleration = vel_trajectory[..., -1, :] - vel_trajectory[..., -2, :]

    future_velocity = vel_trajectory[..., -1, :] + last_acceleration
    vel_trajectory = torch.cat((vel_trajectory, future_velocity[..., None, :]), dim=-2)

    angles = torch.atan2(*(forward.transpose(0, -1))).transpose(0, -1)

    # True difference of angles
    mat_rotZ = axis_angle_rotation("Z", angles)
    vel_mat_rotZ = mat_rotZ[..., 1:, :, :] @ mat_rotZ.transpose(-1, -2)[..., :-1, :, :]
    # repeat the last acceleration (same as the trajectory but in the 3D rotation space)
    last_acc_rotZ = (
        vel_mat_rotZ[..., -1, :, :] @ vel_mat_rotZ.transpose(-1, -2)[..., -2, :, :]
    )
    future_vel_rotZ = vel_mat_rotZ[..., -1, :, :] @ last_acc_rotZ
    vel_mat_rotZ = torch.cat((vel_mat_rotZ, future_vel_rotZ[..., None, :, :]), dim=-3)
    vel_angles = matrix_to_axis_angle(vel_mat_rotZ)[..., 2]

    # Construct the inverse rotation matrix
    rotations_inv = mat_rotZ.transpose(-1, -2)[..., :2, :2]

    poses_local = torch.einsum("...lj,...jk->...lk", poses[..., [0, 1]], rotations_inv)
    poses_local = torch.stack(
        (poses_local[..., 0], poses_local[..., 1], poses[..., 2]), axis=-1
    )

    # stack the xyz joints into feature vectors
    poses_features = rearrange(poses_local, "... joints xyz -> ... (joints xyz)")

    # Rotate the vel_trajectory
    vel_trajectory_local = torch.einsum(
        "...j,...jk->...k", vel_trajectory, rotations_inv
    )

    # Stack things together
    features = group(root_grav_axis, poses_features, vel_angles, vel_trajectory_local)
    return features


def rifke_to_joints(features: Tensor, jointstype="smpljoints") -> Tensor:
    root_grav_axis, poses_features, vel_angles, vel_trajectory_local = ungroup(features)

    # Remove the dummy last angle and integrate the angles
    angles = torch.cumsum(vel_angles[..., :-1], dim=-1)
    # The first angle is zero
    angles = torch.cat((0 * angles[..., [0]], angles), dim=-1)
    rotations = axis_angle_rotation("Z", angles)[..., :2, :2]

    # Get back the poses
    poses_local = rearrange(poses_features, "... (joints xyz) -> ... joints xyz", xyz=3)

    # Rotate the poses
    poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 1]], rotations)
    poses = torch.stack((poses[..., 0], poses[..., 1], poses_local[..., 2]), axis=-1)

    # Rotate the vel_trajectory
    vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local, rotations)
    # Remove the dummy last velocity and integrate the trajectory
    trajectory = torch.cumsum(vel_trajectory[..., :-1, :], dim=-2)
    # The first position is zero
    trajectory = torch.cat((0 * trajectory[..., [0], :], trajectory), dim=-2)

    # Add the root joints (which is still zero)
    poses = torch.cat((0 * poses[..., [0], :], poses), -2)

    # put back the gravity offset
    poses[..., 0, 2] = root_grav_axis

    # Add the trajectory globally
    poses[..., [0, 1]] += trajectory[..., None, :]
    return poses


def group(root_grav_axis, poses_features, vel_angles, vel_trajectory_local):
    # Stack things together
    features = torch.cat(
        (
            root_grav_axis[..., None],
            poses_features,
            vel_angles[..., None],
            vel_trajectory_local,
        ),
        -1,
    )
    return features


def ungroup(features: Tensor) -> tuple[Tensor]:
    # Unbind things
    root_grav_axis = features[..., 0]
    poses_features = features[..., 1:-3]
    vel_angles = features[..., -3]
    vel_trajectory_local = features[..., -2:]
    return root_grav_axis, poses_features, vel_angles, vel_trajectory_local


def get_forward_direction(poses, jointstype="smpljoints"):
    assert jointstype in INFOS
    infos = INFOS[jointstype]
    assert poses.shape[-2] == infos["njoints"]
    RH, LH, RS, LS = infos["RH"], infos["LH"], infos["RS"], infos["LS"]
    across = (
        poses[..., RH, :] - poses[..., LH, :] + poses[..., RS, :] - poses[..., LS, :]
    )
    forward = torch.stack((-across[..., 1], across[..., 0]), axis=-1)
    forward = torch.nn.functional.normalize(forward, dim=-1)
    return forward


def canonicalize_rotation(joints, jointstype="smpljoints"):
    return_np = False
    if isinstance(joints, np.ndarray):
        joints = torch.from_numpy(joints)
        return_np = True

    features = joints_to_rifke(joints, jointstype=jointstype)
    joints_c = rifke_to_joints(features, jointstype=jointstype)
    if return_np:
        joints_c = joints_c.numpy()
    return joints_c
