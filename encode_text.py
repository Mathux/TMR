import os
from omegaconf import DictConfig
import logging
import hydra

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="encode_text")
def encode_text(cfg: DictConfig) -> None:
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name
    text = cfg.text

    import src.prepare  # noqa
    import torch
    import numpy as np
    from src.config import read_config
    from src.load import load_model_from_cfg
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything
    from src.data.collate import collate_x_dict

    cfg = read_config(run_dir)

    logger.info("Loading the text model")
    text_model = instantiate(cfg.data.text_to_token_emb, device=device)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    seed_everything(cfg.seed)
    with torch.inference_mode():
        text_x_dict = collate_x_dict(text_model([text]))
        latent = model.encode(text_x_dict, sample_mean=True)[0]
        latent = latent.cpu().numpy()

    fname = text.lower().replace(" ", "_") + ".npy"

    output_folder = os.path.join(run_dir, "encoded")
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, fname)

    np.save(path, latent)
    logger.info(f"Encoding done, latent saved in:\n{path}")


if __name__ == "__main__":
    encode_text()
