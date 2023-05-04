import os
import orjson
import torch
import numpy as np
from model import TMR_textencoder

EMBS = "data/unit_motion_embs"


def load_json(path):
    with open(path, "rb") as ff:
        return orjson.loads(ff.read())


def load_keyids(split):
    path = os.path.join(EMBS, f"{split}.keyids")
    with open(path) as ff:
        keyids = np.array([x.strip() for x in ff.readlines()])
    return keyids


def load_keyids_splits(splits):
    return {
        split: load_keyids(split)
        for split in splits
    }


def load_unit_motion_embs(split, device):
    path = os.path.join(EMBS, f"{split}_motion_embs_unit.npy")
    tensor = torch.from_numpy(np.load(path)).to(device)
    return tensor


def load_unit_motion_embs_splits(splits, device):
    return {
        split: load_unit_motion_embs(split, device)
        for split in splits
    }


def load_model(device):
    text_params = {
        'latent_dim': 256, 'ff_size': 1024, 'num_layers': 6, 'num_heads': 4,
        'activation': 'gelu', 'modelpath': 'distilbert-base-uncased'
    }
    "unit_motion_embs"
    model = TMR_textencoder(**text_params)
    state_dict = torch.load("data/textencoder.pt", map_location=device)
    # load values for the transformer only
    model.load_state_dict(state_dict, strict=False)
    model = model.eval()
    return model
