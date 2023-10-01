import logging
import torch
import warnings

logger = logging.getLogger("torch.distributed.nn.jit.instantiator")
logger.setLevel(logging.ERROR)


warnings.filterwarnings(
    "ignore", "The PyTorch API of nested tensors is in prototype stage*"
)

warnings.filterwarnings("ignore", "Converting mask without torch.bool dtype to bool*")

torch.set_float32_matmul_precision("high")
