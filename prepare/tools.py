import os
from glob import glob
from tqdm import tqdm


def loop_amass(
    base_folder,
    new_base_folder,
    ext=".npz",
    newext=".npz",
    force_redo=False,
    exclude=None,
):
    match_str = f"**/*{ext}"

    for motion_file in tqdm(glob(match_str, root_dir=base_folder, recursive=True)):
        if exclude and exclude in motion_file:
            continue

        motion_path = os.path.join(base_folder, motion_file)

        if motion_path.endswith("shape.npz"):
            continue

        new_motion_path = os.path.join(
            new_base_folder, motion_file.replace(ext, newext)
        )
        if not force_redo and os.path.exists(new_motion_path):
            continue

        new_folder = os.path.split(new_motion_path)[0]
        os.makedirs(new_folder, exist_ok=True)

        yield motion_path, new_motion_path
