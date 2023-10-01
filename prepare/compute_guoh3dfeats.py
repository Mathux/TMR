import os
import logging
import hydra
from omegaconf import DictConfig

import numpy as np

logger = logging.getLogger(__name__)


def extract_h3d(feats):
    from einops import unpack

    root_data, ric_data, rot_data, local_vel, feet_l, feet_r = unpack(
        feats, [[4], [63], [126], [66], [2], [2]], "i *"
    )
    return root_data, ric_data, rot_data, local_vel, feet_l, feet_r


def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


@hydra.main(
    config_path="../configs", config_name="compute_guoh3dfeats", version_base="1.3"
)
def compute_guoh3dfeats(cfg: DictConfig):
    base_folder = cfg.base_folder
    output_folder = cfg.output_folder
    force_redo = cfg.force_redo

    from src.guofeats import joints_to_guofeats
    from .tools import loop_amass

    output_folder_M = os.path.join(output_folder, "M")

    print("Get h3d features from Guo et al.")
    print("The processed motions will be stored in this folder:")
    print(output_folder)

    iterator = loop_amass(
        base_folder, output_folder, ext=".npy", newext=".npy", force_redo=force_redo
    )

    for motion_path, new_motion_path in iterator:
        joints = np.load(motion_path)

        if "humanact12" not in motion_path:
            # This is because the authors of HumanML3D
            # save the motions by swapping Y and Z (det = -1)
            # which is not a proper rotation (det = 1)
            # so we should invert x, to make it a rotation
            # that is why the authors use "data[..., 0] *= -1" inside the "if"
            # before swapping left/right
            # https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb
            joints[..., 0] *= -1
            # the humanact12 motions are normally saved correctly, no need to swap again
            # (but in fact this may not be true and the orignal H3D features
            # corresponding to HumanAct12 appears to be left/right flipped..)
            # At least we are compatible with previous work :/

        joints_m = swap_left_right(joints)

        # apply transformation
        try:
            features = joints_to_guofeats(joints)
            features_m = joints_to_guofeats(joints_m)
        except (IndexError, ValueError):
            # The sequence should be only 1 frame long
            # so we cannot compute features (which involve velocities etc)
            assert len(joints) == 1
            continue
        # save the features
        np.save(new_motion_path, features)

        # save the mirrored features as well
        new_motion_path_M = new_motion_path.replace(output_folder, output_folder_M)
        os.makedirs(os.path.split(new_motion_path_M)[0], exist_ok=True)
        np.save(new_motion_path_M, features_m)


if __name__ == "__main__":
    compute_guoh3dfeats()
