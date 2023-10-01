import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="text_embeddings", version_base="1.3")
def text_embeddings(cfg: DictConfig):
    device = cfg.device

    import src.prepare  # noqa
    from src.data.text import save_token_embeddings, save_sent_embeddings

    # Compute token embeddings
    modelname = cfg.data.text_to_token_emb.modelname
    logger.info(f"Compute token embeddings for {modelname}")
    path = cfg.data.text_to_token_emb.path
    save_token_embeddings(path, modelname=modelname, device=device)

    # Compute sent embeddings
    modelname = cfg.data.text_to_sent_emb.modelname
    logger.info(f"Compute sentence embeddings for {modelname}")
    path = cfg.data.text_to_sent_emb.path
    save_sent_embeddings(path, modelname=modelname, device=device)


if __name__ == "__main__":
    text_embeddings()
