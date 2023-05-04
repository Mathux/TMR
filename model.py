from typing import List
import torch.nn as nn
import os

import torch
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers import logging
from torch.nn.functional import normalize


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:x.shape[0], :]


class TMR_textencoder(nn.Module):
    def __init__(self, modelpath: str, latent_dim: int, ff_size: int,
                 num_layers: int, num_heads: int, activation: str, **kwargs) -> None:
        super().__init__()

        logging.set_verbosity_error()

        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)
        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size

        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.text_encoded_dim, latent_dim)
        )

        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(latent_dim)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=0.0,
                                                             activation=activation)
        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer,
            num_layers=num_layers
        )

    def get_last_hidden_state(self, texts: List[str],
                              return_mask: bool = False):
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool)

    def forward(self, texts: List[str]) -> Tensor:
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True)

        x = self.projection(text_encoded)
        bs, nframes, _ = x.shape
        # bs, nframes, totjoints, nfeats = x.shape
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
        logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

        # adding the distribution tokens for all sequences
        xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

        # create a bigger mask, to allow attend to mu and logvar
        token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        # only mu for inference
        mu = final[0]
        return mu

    # compute score for retrieval
    def compute_scores(self, texts, unit_embs=None, embs=None):
        # not both empty
        assert not (unit_embs is None and embs is None)
        # not both filled
        assert not (unit_embs is not None and embs is not None)

        output_str = False
        # if one input, squeeze the output
        if isinstance(texts, str):
            texts = [texts]
            output_str = True

        # compute unit_embs from embs if not given
        if embs is not None:
            unit_embs = normalize(embs)

        with torch.no_grad():
            latent_unit_texts = normalize(self(texts))
            # compute cosine similarity between 0 and 1
            scores = (unit_embs @ latent_unit_texts.T).T/2 + 0.5
            scores = scores.cpu().numpy()

        if output_str:
            scores = scores[0]

        return scores
