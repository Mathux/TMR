import os
from omegaconf import DictConfig
import logging
import hydra
import json
from hydra.core.hydra_config import HydraConfig


logger = logging.getLogger(__name__)


def x_dict_to_device(x_dict, device):
    import torch

    for key, val in x_dict.items():
        if isinstance(val, torch.Tensor):
            x_dict[key] = val.to(device)
    return x_dict


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))


@hydra.main(version_base=None, config_path="configs", config_name="encode_dataset")
def encode_dataset(cfg: DictConfig) -> None:
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name
    cfg_data = cfg.data

    choices = HydraConfig.get().runtime.choices
    data_name = choices.data

    import src.prepare  # noqa
    import torch
    import numpy as np
    from src.config import read_config
    from src.load import load_model_from_cfg
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything

    cfg = read_config(run_dir)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    save_dir = os.path.join(run_dir, "latents")
    os.makedirs(save_dir, exist_ok=True)

    dataset = instantiate(cfg_data, split="all")
    dataloader = instantiate(
        cfg.dataloader,
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )
    seed_everything(cfg.seed)

    all_latents = []
    all_keyids = []

    with torch.inference_mode():
        for batch in dataloader:
            motion_x_dict = batch["motion_x_dict"]
            x_dict_to_device(motion_x_dict, device)
            latents = model.encode(motion_x_dict, sample_mean=True)
            all_latents.append(latents.cpu().numpy())
            keyids = batch["keyid"]
            all_keyids.extend(keyids)

    latents = np.concatenate(all_latents)
    path = os.path.join(save_dir, f"{data_name}_all.npy")
    logger.info(f"Encoding the latents of all the splits in {path}")
    np.save(path, latents)

    path_unit = os.path.join(save_dir, f"{data_name}_all_unit.npy")
    logger.info(f"Encoding the unit latents of all the splits in {path_unit}")

    unit_latents = latents / np.linalg.norm(latents, axis=-1)[:, None]
    np.save(path_unit, unit_latents)

    # Writing the correspondance
    logger.info("Writing the correspondance files")
    keyids_index_path = os.path.join(save_dir, f"{data_name}_keyids_index_all.json")
    index_keyids_path = os.path.join(save_dir, f"{data_name}_index_keyids_all.json")

    keyids_index = {x: i for i, x in enumerate(all_keyids)}
    index_keyids = {i: x for i, x in enumerate(all_keyids)}

    write_json(keyids_index, keyids_index_path)
    write_json(index_keyids, index_keyids_path)


if __name__ == "__main__":
    encode_dataset()
