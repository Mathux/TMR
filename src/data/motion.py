import os
import torch
import numpy as np


class AMASSMotionLoader:
    def __init__(
        self, base_dir, fps, normalizer=None, disable: bool = False, nfeats=None
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.normalizer = normalizer
        self.disable = disable
        self.nfeats = nfeats

    def __call__(self, path, start, end):
        if self.disable:
            return {"x": path, "length": int(self.fps * (end - start))}

        begin = int(start * self.fps)
        end = int(end * self.fps)
        if path not in self.motions:
            motion_path = os.path.join(self.base_dir, path + ".npy")
            motion = np.load(motion_path)
            motion = torch.from_numpy(motion).to(torch.float)
            if self.normalizer is not None:
                motion = self.normalizer(motion)
            self.motions[path] = motion

        motion = self.motions[path][begin:end]
        x_dict = {"x": motion, "length": len(motion)}
        return x_dict


class Normalizer:
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        self.base_dir = base_dir
        self.mean_path = os.path.join(base_dir, "mean.pt")
        self.std_path = os.path.join(base_dir, "std.pt")
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        self.mean = torch.load(self.mean_path)
        self.std = torch.load(self.std_path)

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        torch.save(mean, self.mean_path)
        torch.save(std, self.std_path)

    def __call__(self, x):
        if self.disable:
            return x
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        x = x * (self.std + self.eps) + self.mean
        return x
