import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="motion_stats", version_base="1.3")
def motion_stats(cfg: DictConfig):
    logger.info("Computing motion stats")
    import src.prepare  # noqa

    train_dataset = instantiate(cfg.data, split="train")
    import torch

    feats = torch.cat([x["motion_x_dict"]["x"] for x in tqdm(train_dataset)])
    mean = feats.mean(0)
    std = feats.std(0)

    normalizer = train_dataset.motion_loader.normalizer
    logger.info(f"Saving them in {normalizer.base_dir}")
    normalizer.save(mean, std)


if __name__ == "__main__":
    motion_stats()
