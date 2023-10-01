import logging

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


logger = logging.getLogger(__name__)


class ProgressLogger(Callback):
    def __init__(self, precision: int = 2):
        self.precision = precision

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs):
        logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule, **kwargs):
        logger.info("Training done")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ):
        if trainer.sanity_checking:
            logger.info("Sanity checking ok.")

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ):
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        metrics_str = []

        losses_dict = trainer.callback_metrics

        def is_contrastive_metrics(x):
            return "t2m" in x or "m2t" in x

        losses_to_print = [
            x
            for x in losses_dict.keys()
            for y in [x.split("_")]
            if len(y) == 3
            and y[2] == "epoch"
            and (
                y[1] in pl_module.lmd or y[1] == "loss" or is_contrastive_metrics(y[1])
            )
        ]

        # Natual order for contrastive
        letters = "0123456789"
        mapping = str.maketrans(letters, letters[::-1])

        def sort_losses(x):
            split, name, epoch_step = x.split("_")
            if is_contrastive_metrics(x):
                # put them at the end
                name = "a" + name.translate(mapping)
            return (name, split)

        losses_to_print = sorted(losses_to_print, key=sort_losses, reverse=True)
        for metric_name in losses_to_print:
            split, name, _ = metric_name.split("_")

            metric = losses_dict[metric_name].item()

            if is_contrastive_metrics(metric_name):
                if "len" in metric_name:
                    metric = str(int(metric))
                elif "MedR" in metric_name:
                    metric = str(int(metric * 100) / 100) + "%"
                else:
                    metric = str(int(metric * 100) / 100) + "%"
            else:
                metric = metric_format.format(metric)

            if split == "train":
                mname = name
            else:
                mname = f"v_{name}"

            metric = f"{mname} {metric}"
            metrics_str.append(metric)

        if len(metrics_str) == 0:
            return

        line = line + ": " + "  ".join(metrics_str)
        logger.info(line)
