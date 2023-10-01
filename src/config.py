import os
import json
from omegaconf import DictConfig, OmegaConf


def save_config(cfg: DictConfig) -> str:
    path = os.path.join(cfg.run_dir, "config.json")
    config = OmegaConf.to_container(cfg, resolve=True)
    with open(path, "w") as f:
        string = json.dumps(config, indent=4)
        f.write(string)
    return path


def read_config(run_dir: str, return_json=False) -> DictConfig:
    path = os.path.join(run_dir, "config.json")
    with open(path, "r") as f:
        config = json.load(f)
    if return_json:
        return config
    cfg = OmegaConf.create(config)
    cfg.run_dir = run_dir
    return cfg
