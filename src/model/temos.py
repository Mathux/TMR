from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule

from src.model.losses import KLLoss


def length_to_mask(length: List[int], device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


class TEMOS(LightningModule):
    r"""TEMOS: Generating diverse human motions
    from textual descriptions
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/temos

    Args:
        motion_encoder: a module to encode the input motion features in the latent space (required).
        text_encoder: a module to encode the text embeddings in the latent space (required).
        motion_decoder: a module to decode the latent vector into motion features (required).
        vae: a boolean to make the model probabilistic (required).
        fact: a scaling factor for sampling the VAE (optional).
        sample_mean: sample the mean vector instead of random sampling (optional).
        lmd: dictionary of losses weights (optional).
        lr: learninig rate for the optimizer (optional).
    """

    def __init__(
        self,
        motion_encoder: nn.Module,
        text_encoder: nn.Module,
        motion_decoder: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5},
        lr: float = 1e-4,
    ) -> None:
        super().__init__()

        self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder
        self.motion_decoder = motion_decoder

        # sampling parameters
        self.vae = vae
        self.fact = fact if fact is not None else 1.0
        self.sample_mean = sample_mean

        # losses
        self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()

        # lambda weighting for the losses
        self.lmd = lmd
        self.lr = lr

    def configure_optimizers(self) -> None:
        return {"optimizer": torch.optim.AdamW(lr=self.lr, params=self.parameters())}

    def _find_encoder(self, inputs, modality):
        assert modality in ["text", "motion", "auto"]

        if modality == "text":
            return self.text_encoder
        elif modality == "motion":
            return self.motion_encoder

        m_nfeats = self.motion_encoder.nfeats
        t_nfeats = self.text_encoder.nfeats

        if m_nfeats == t_nfeats:
            raise ValueError(
                "Cannot automatically find the encoder, as they share the same input space."
            )

        nfeats = inputs["x"].shape[-1]
        if nfeats == m_nfeats:
            return self.motion_encoder
        elif nfeats == t_nfeats:
            return self.text_encoder
        else:
            raise ValueError("The inputs is not recognized.")

    def encode(
        self,
        inputs,
        modality: str = "auto",
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_distribution: bool = False,
    ):
        sample_mean = self.sample_mean if sample_mean is None else sample_mean
        fact = self.fact if fact is None else fact

        # Encode the inputs
        encoder = self._find_encoder(inputs, modality)
        encoded = encoder(inputs)

        # Sampling
        if self.vae:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + fact * eps * std
        else:
            dists = None
            (latent_vectors,) = encoded.unbind(1)

        if return_distribution:
            return latent_vectors, dists

        return latent_vectors

    def decode(
        self,
        latent_vectors: Tensor,
        lengths: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
    ):
        mask = mask if mask is not None else length_to_mask(lengths, device=self.device)
        z_dict = {"z": latent_vectors, "mask": mask}
        motions = self.motion_decoder(z_dict)
        return motions

    # Forward: X => motions
    def forward(
        self,
        inputs,
        lengths: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_all: bool = False,
    ) -> List[Tensor]:
        # Encoding the inputs and sampling if needed
        latent_vectors, distributions = self.encode(
            inputs, sample_mean=sample_mean, fact=fact, return_distribution=True
        )
        # Decoding the latent vector: generating motions
        motions = self.decode(latent_vectors, lengths, mask)

        if return_all:
            return motions, latent_vectors, distributions

        return motions

    def compute_loss(self, batch: Dict) -> Dict:
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]

        mask = motion_x_dict["mask"]
        ref_motions = motion_x_dict["x"]

        # text -> motion
        t_motions, t_latents, t_dists = self(text_x_dict, mask=mask, return_all=True)

        # motion -> motion
        m_motions, m_latents, m_dists = self(motion_x_dict, mask=mask, return_all=True)

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_motions, ref_motions) # text -> motion
            + self.reconstruction_loss_fn(m_motions, ref_motions) # motion -> motion
        )
        # fmt: on

        # VAE losses
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                self.kl_loss_fn(t_dists, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists)  # motion_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # motion
                + self.kl_loss_fn(t_dists, ref_dists)  # text
            )

        # Latent manifold loss
        losses["latent"] = self.latent_loss_fn(t_latents, m_latents)

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )
        return losses

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])
        losses = self.compute_loss(batch)

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )
        return losses["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])
        losses = self.compute_loss(batch)

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )
        return losses["loss"]
