import os
import numpy as np
import orjson
import codecs as cs
import torch


def load_json(json_path):
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


def load_unit_embeddings(run_dir, dataset, device="cpu"):
    save_dir = os.path.join(run_dir, "latents")
    unit_emb_path = os.path.join(save_dir, f"{dataset}_all_unit.npy")
    motion_embs = torch.from_numpy(np.load(unit_emb_path)).to(device)

    # Loading the correspondance
    keyids_index = load_json(os.path.join(save_dir, f"{dataset}_keyids_index_all.json"))
    index_keyids = load_json(os.path.join(save_dir, f"{dataset}_index_keyids_all.json"))

    return motion_embs, keyids_index, index_keyids


def load_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_splits(dataset, splits=["test", "all"]):
    path = f"datasets/annotations/{dataset}"
    return {split: load_split(path, split) for split in splits}
