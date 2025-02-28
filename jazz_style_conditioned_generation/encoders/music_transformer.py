#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Music Transformer module, taken from https://github.com/gwinndr/MusicTransformer-Pytorch"""

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from transformers.utils import ModelOutput

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.encoders.rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR


@dataclass
class MusicTransformerOutput(ModelOutput):
    """For consistency with the `transformers` API"""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class MusicTransformer(nn.Module):
    """Music Transformer reproduction from https://arxiv.org/abs/1809.04281."""

    def __init__(
            self,
            tokenizer,
            n_layers=6,
            num_heads=8,
            d_model=512,
            dim_feedforward=1024,
            dropout=0.1,
            max_sequence=2048,
            rpr=False
    ):
        super(MusicTransformer, self).__init__()

        self.tokenizer = tokenizer
        self.dummy = DummyDecoder()

        self.nlayers = n_layers
        self.nhead = num_heads
        self.d_model = d_model
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.max_seq = max_sequence
        self.rpr = rpr

        # Input embedding
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.d_model)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)
        # Base transformer
        if not self.rpr:
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            self.transformer = nn.Transformer(
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.nlayers,
                num_decoder_layers=0,
                dropout=self.dropout,
                # activation=self.ff_activ,
                dim_feedforward=self.d_ff,
                custom_decoder=self.dummy
            )
        # RPR Transformer
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(
                self.d_model,
                self.nhead,
                self.d_ff,
                self.dropout,
                er_len=self.max_seq
            )
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.nlayers,
                num_decoder_layers=0,
                dropout=self.dropout,
                # activation=self.ff_activ,
                dim_feedforward=self.d_ff,
                custom_decoder=self.dummy,
                custom_encoder=encoder
            )
        # Final output is a softmaxed linear layer
        self.Wout = nn.Linear(self.d_model, self.tokenizer.vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        # Calculate loss internally within the class, as in `transformers`
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def forward(self, input_ids, labels, attention_mask):
        """Takes an input sequence and outputs predictions using a sequence to sequence method."""
        # Create causal mask
        mask = self.transformer.generate_square_subsequent_mask(input_ids.shape[1]).to(utils.DEVICE)
        x = self.embedding(input_ids)
        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(
            src=x,
            tgt=x,
            src_mask=mask,
            src_key_padding_mask=attention_mask  # masks PAD tokens (i.e., to ensure we have sequence length)
        )
        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1, 0, 2)
        # Compute logits from FC layer
        logits = self.Wout(x_out)  # No softmax as nn.CrossEntropyLoss computes it for us
        # Flatten logits to (sequence_len * batch_size, vocab_size)
        y = logits.reshape(logits.shape[0] * logits.shape[1], -1).to(torch.float32)
        # Flatten targets to (sequence_len * batch_size)
        t = labels.flatten().to(torch.int64)
        # Compute cross-entropy loss
        loss = self.loss(y, t)
        # Returns output in the same format as transformers
        return MusicTransformerOutput(
            loss=loss,
            logits=logits
        )

    def generate(self, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0):
        """Generates midi given a primer sample."""
        assert not self.training, "Cannot generate while in training mode"
        print("Generating sequence of max length:", target_seq_length)

        gen_seq = torch.full(
            (1, target_seq_length),
            self.tokenizer["PAD_None"],
            dtype=torch.long,
            device=utils.DEVICE
        )

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(torch.long).to(utils.DEVICE)

        # print("primer:",primer)
        # print(gen_seq)
        cur_i = num_primer
        while cur_i < target_seq_length:
            # gen_seq_batch     = gen_seq.clone()
            # TODO: this is broken
            y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :self.tokenizer["EOS_None"]]
            token_probs = y[:, cur_i - 1, :]

            if beam == 0:
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0, 1)

            if beam_ran <= beam_chance:
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // self.tokenizer.vocab_size
                beam_cols = top_i % self.tokenizer.vocab_size

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                # print("next token:",next_token)
                gen_seq[:, cur_i] = next_token

                # Let the transformer decide to end if it wants to
                if next_token == self.tokenizer["EOS_None"]:
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break

            cur_i += 1
            if cur_i % 50 == 0:
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]


class DummyDecoder(nn.Module):
    """A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only architecture"""

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, **_):
        """Returns the input (memory)"""
        return memory


class PositionalEncoding(nn.Module):
    """Positional encoding, from https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = utils.MAX_SEQUENCE_LENGTH
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    from miditok import REMI
    from jazz_style_conditioned_generation.data.dataloader import create_padding_mask

    token_factory = REMI()
    mt = MusicTransformer(token_factory, max_sequence=2048).to(utils.DEVICE)
    dummy_tensor = torch.randint(0, 100, (4, 2049))

    inp = dummy_tensor[:, :-1].to(utils.DEVICE)
    targ = dummy_tensor[:, 1:].to(utils.DEVICE)
    padding_mask = create_padding_mask(dummy_tensor, token_factory.pad_token_id)[:, :-1].to(utils.DEVICE)

    out = mt(inp, targ, padding_mask)
    print(f'Dummy loss: {out.loss.item()}')
