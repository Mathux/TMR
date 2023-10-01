# from lightning_fabric/loggers/csv_logs.py
# of lightning_fabric version 2.04

# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union

from torch import Tensor

from lightning_fabric.loggers.logger import Logger, rank_zero_experiment
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.logger import _add_prefix
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.types import _PATH

log = logging.getLogger(__name__)


class CSVLogger(Logger):
    r"""Log to the local file system in CSV format.

    Logs are saved to ``os.path.join(root_dir, name)``.

    Args:
        root_dir: The root directory in which all your experiments with different names and versions will be stored.
        name: Experiment name. Defaults to ``'lightning_logs'``.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

    Example::

        from lightning_fabric.loggers import CSVLogger

        logger = CSVLogger("path/to/logs/root", name="my_model")
        logger.log_metrics({"loss": 0.235, "acc": 0.75})
        logger.finalize("success")
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        root_dir: _PATH,
        name: str = "lightning_logs",
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__()
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._name = name or ""
        self._prefix = prefix
        self._fs = get_filesystem(root_dir)
        self._experiment: Optional[_ExperimentWriter] = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.
        """
        return self._name

    @property
    def version(self) -> str:
        return ""

    @property
    def root_dir(self) -> str:
        """Gets the save directory where the versioned CSV experiments are saved."""
        return self._root_dir

    @property
    def log_dir(self) -> str:
        """The log directory for this run."""
        # create a pseudo standard path
        return os.path.join(self.root_dir, self.name)

    @property
    @rank_zero_experiment
    def experiment(self) -> "_ExperimentWriter":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code, do the
        following.

        Example::

            self.logger.experiment.some_experiment_writer_function()
        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = _ExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        raise NotImplementedError(
            "The `CSVLogger` does not yet support logging hyperparameters."
        )

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, Union[Tensor, float]], step: Optional[int] = None
    ) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log_metrics(metrics, step)
        if step is not None and (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._experiment is None:
            # When using multiprocessing, finalize() should be a no-op on the main process, as no experiment has been
            # initialized there
            return
        self.save()


class _ExperimentWriter:
    r"""Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs
    """

    NAME_METRICS_FILE = "metrics.csv"

    def __init__(self, log_dir: str) -> None:
        self.metrics: List[Dict[str, float]] = []

        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        self._fs.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            # Read previous logs
            if os.path.exists(self.metrics_file_path):
                with self._fs.open(self.metrics_file_path, "r") as f:
                    reader = csv.DictReader(f)
                    self.metrics = [x for x in reader]

    def log_metrics(
        self, metrics_dict: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return

        last_m = {}
        for m in self.metrics:
            last_m.update(m)
        metrics_keys = list(last_m.keys())

        with self._fs.open(self.metrics_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_keys)
            writer.writeheader()
            writer.writerows(self.metrics)
