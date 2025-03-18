#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Music Transformer module, taken from https://github.com/gwinndr/MusicTransformer-Pytorch"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from jazz_style_conditioned_generation import utils, metrics
from jazz_style_conditioned_generation.data.dataloader import create_padding_mask
from jazz_style_conditioned_generation.encoders.rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR

SULUN_2022_CONFIG = dict(
    rpr=True,
    n_layers=20,
    num_heads=16,
    d_model=768,
    dim_feedforward=3072,
    dropout=0.1,
    max_sequence=1216
)
ROW_2024_MODEL_1_CONFIG = dict(
    rpr=True,
    n_layers=2,
    dim_feedforward=256,
    num_heads=8,
    d_model=64,
)
ROW_2024_MODEL_2_CONFIG = dict(
    rpr=True,
    n_layers=4,
    dim_feedforward=512,
    num_heads=8,
    d_model=128
)
DEFAULT_CONFIG = dict(
    n_layers=6,
    num_heads=8,
    d_model=512,
    dim_feedforward=1024,
    dropout=0.1,
)


def top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """ Copied from transformers"""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[:, :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


@dataclass
class MusicTransformerOutput(ModelOutput):
    """For consistency with the `transformers` API"""
    loss: Optional[torch.FloatTensor] = None
    decoded_loss: Optional[torch.FloatTensor] = None
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
            n_layers: int = 6,  # Sulun == 20
            num_heads: int = 8,  # Sulun === 16
            d_model: int = 512,  # Sulun == 768
            dim_feedforward: int = 1024,  # Sulun == 3072
            dropout: float = 0.1,  # Sulun == 0.1
            max_sequence: int = utils.MAX_SEQUENCE_LENGTH,
            rpr: bool = False
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
            encoder_norm = nn.modules.normalization.LayerNorm(self.d_model)
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

    def forward(
            self,
            input_ids: torch.tensor,
            labels: torch.tensor,
            attention_mask: torch.tensor
    ) -> MusicTransformerOutput:
        """Takes an input sequence and outputs predictions using a sequence to sequence method."""
        # Create causal mask
        mask = self.transformer.generate_square_subsequent_mask(input_ids.shape[1]).to(input_ids.device)
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
        # Compute logits from FC layer: shape (batch_size, seq_len, vocab_size)
        logits = self.Wout(x_out)  # No softmax as nn.CrossEntropyLoss computes it for us
        # Compute loss: vanilla first (i.e., without decoding BPE tokens), then with decoding
        vanilla_loss = metrics._cross_entropy_loss(logits, labels, self.tokenizer)
        decoded_loss = metrics.cross_entropy_loss(logits, labels, self.tokenizer, len(self.tokenizer._vocab_base))
        # Returns output in the same format as transformers
        return MusicTransformerOutput(loss=vanilla_loss, logits=logits, decoded_loss=decoded_loss)

    def generate(
            self,
            primer: torch.tensor = None,
            target_seq_length: int = utils.MAX_SEQUENCE_LENGTH,
            top_p: float = 1.,
            top_k: int = 0
    ):
        """Generates midi given a primer sample with nucleus sampling."""
        assert not self.training, "Cannot generate while in training mode"
        if primer.dim() > 1:
            raise ValueError(f"Expected a tensor with 1 dimension for generation, but got {primer.dim()} dimensions!")

        # Create an empty array with the target sequence length
        gen_seq = torch.full(
            (1, target_seq_length),
            self.tokenizer["PAD_None"],
            dtype=torch.long,
            device=utils.DEVICE
        )
        # This counter keeps track of where we are in the sequence
        cur_i = 1
        # Add the BOS token at the start of the sequence, if it is not present already
        if gen_seq[:, 0] != self.tokenizer["BOS_None"]:
            gen_seq[:, 0] = self.tokenizer["BOS_None"]
        # Fill in the empty array with our primer sequence if we've provided this
        if primer is not None:
            num_primer = len(primer) + 1
            gen_seq[:, 1: num_primer] = primer.type(torch.long).to(utils.DEVICE)
            cur_i = num_primer
        # Keep iterating until we hit the desired sequence length
        while cur_i < target_seq_length:
            # Unpack everything from the currently generated sequence
            input_ids = gen_seq[:, :cur_i]
            labels = gen_seq[:, 1: cur_i + 1]
            attention_mask = create_padding_mask(input_ids, self.tokenizer["PAD_None"])
            # Through the model to get the logits: shape (1, sequence_length, vocab_size)
            logits = self.forward(input_ids, labels, attention_mask).logits
            # This gets the probabilities for the next token: shape (1, vocab_size)
            token_probs = logits[:, cur_i - 1, :]
            # Filter to get the top-k and top-p
            # Elements not in top-k/top-p are replaced with 0
            topk_topp = top_k_top_p_filtering(token_probs, top_k, top_p, filter_value=0.)
            # Apply the softmax
            smaxed = self.softmax(topk_topp)
            # Create the distribution
            # TODO: somehow this can end up all NaNs?
            dist = torch.distributions.Categorical(probs=smaxed)
            # Get the next token by sampling from the distribution
            # Elements not in top-k/top-p have probability 0
            next_token = dist.sample()
            # Sanity check, we shouldn't predict an element with probability of 0.
            assert smaxed[:, next_token].item() != 0.
            # Add the next token into the sequence
            gen_seq[:, cur_i] = next_token
            # Increment the counter
            cur_i += 1
            # Let the transformer decide to end the sequence early if it wants to
            if next_token == self.tokenizer["EOS_None"] or cur_i >= target_seq_length:
                break
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
    from jazz_style_conditioned_generation.data.tokenizer import load_tokenizer, train_tokenizer
    from jazz_style_conditioned_generation.data.dataloader import DatasetMIDIConditioned

    # Create a tokenizer and train with a small number of tracks
    midis = utils.get_data_files_with_ext("data/raw", "**/*.mid")[:50]
    tokenizer = load_tokenizer()
    train_tokenizer(tokenizer, midis, vocab_size=1000)

    # Create the dataset + loader
    dataset = DatasetMIDIConditioned(
        tokenizer, midis, do_conditioning=False, do_augmentation=False, max_seq_len=utils.MAX_SEQUENCE_LENGTH
    )
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, drop_last=False, batch_size=2)

    # Create the model
    mt = MusicTransformer(tokenizer=tokenizer, **DEFAULT_CONFIG).to(utils.DEVICE)

    # Get an input batch
    batch1 = next(iter(loader))
    inp, targ = batch1["input_ids"].to(utils.DEVICE), batch1["labels"].to(utils.DEVICE)
    padding_mask = create_padding_mask(inp, tokenizer.pad_token_id).to(utils.DEVICE)

    # Pass through the model
    out = mt(inp, targ, padding_mask)
    print(f'Dummy loss: {out.loss.item()}')

    # Do some generation with the dummy tensor
    mt.eval()
    gen = mt.generate(primer=inp[0, :10].squeeze(0))

    # Convert back to a score
    retok = tokenizer(gen)
    print(f'Dummy generation: {retok}')

# This converts the ID for a learned byte to a list of IDs of the actual tokens
# token_mapping = {
#     self.tokenizer.vocab_model[byt]: [self.tokenizer[t] for t in token_list]
#     for byt, token_list in self.tokenizer._vocab_learned_bytes_to_tokens.items()
# }
# batch_results = []
# for item in logits:
#     item_results = []
#     for step in item:
#         logits_at_step = {id_: [] for id_ in self.tokenizer.vocab.values()}
#         for learned_byte_id, logit in enumerate(step.tolist()):
#             for nonbpe_byte_id in token_mapping[learned_byte_id]:
#                 logits_at_step[nonbpe_byte_id].append(logit)
#         logits_summed = torch.tensor([sum(v) for v in logits_at_step.values()])
#         assert logits_summed.size(0) == len(self.tokenizer.vocab)
#         item_results.append(logits_summed)
#     batch_results.append(torch.stack(item_results))
# logits_decoded = torch.stack(batch_results)
# # Simply take the first token for each BPE index
# labels_decoded = torch.tensor([[token_mapping[i.item()][0] for i in batch] for batch in labels])
# # Flatten everything
# y = logits_decoded.reshape(logits_decoded.shape[0] * logits_decoded.shape[1], -1).to(torch.float32)
# t = labels_decoded.flatten().to(torch.int64)
# loss1 = self.loss(y, t)
# print(loss, loss1)


# for item_logits, item_labels in zip(logits, labels):
#     item_res = []
#     for logit, label in zip(item_logits, item_labels):
#         decoded_label = token_mapping[label.item()]
#         res_at_label = [[[] for _ in range(len(self.tokenizer.vocab.values()))] for _ in range(len(decoded_label))]
#         for log_byte_id, log in enumerate(logit.tolist()):
#             log_token_ids = token_mapping[log_byte_id]
#             if len(log_token_ids) < len(decoded_label):
#                 log_token_ids.extend([log_token_ids[-1] for _ in range(len(decoded_label) - len(log_token-ids))])
#             else:
#                 log_token_ids = log_token_ids[:len(decoded_label)]
#             assert len(log_token_ids) == len(decoded_label)
#             for idx, id_ in enumerate(log_token_ids):
#                 res_at_label[idx][id_].append(log)
#
#         break
#     break
