#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Music Transformer module, adapted from https://github.com/gwinndr/MusicTransformer-Pytorch"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.utils import ModelOutput

from jazz_style_conditioned_generation import utils
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

DEFAULT_TOP_P = 0.7
DEFAULT_TEMPERATURE = 1.3


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
            max_seq_len: int = utils.MAX_SEQUENCE_LENGTH,
            rpr: bool = False,
            dim_condition: int = 0
    ):
        super(MusicTransformer, self).__init__()

        self.tokenizer = tokenizer
        self.dummy = DummyDecoder()

        self.nlayers = n_layers
        self.nhead = num_heads
        self.d_model = d_model
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.rpr = rpr

        # Input embedding
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.d_model)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq_len)
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
                custom_decoder=self.dummy,
                batch_first=True
            )
        # RPR Transformer
        else:
            encoder_norm = nn.modules.normalization.LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(
                self.d_model,
                self.nhead,
                self.d_ff,
                self.dropout,
                er_len=self.max_seq_len
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
                custom_encoder=encoder,
                batch_first=True
            )
        # Final output is a softmaxed linear layer
        self.Wout = nn.Linear(self.d_model, self.tokenizer.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor,
            attention_mask: torch.Tensor,
            condition_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """Takes an input sequence and outputs predictions using a sequence to sequence method, returns raw logits."""
        # Create causal mask
        causal_mask = self.transformer.generate_square_subsequent_mask(input_ids.shape[1]).to(input_ids.device)
        x = self.embedding(input_ids)
        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        # Back to (batch_size, max_seq, d_model)
        x = x.permute(1, 0, 2)
        # Since there are no true decoder layers, the tgt is unused
        x_out = self.transformer(
            src=x,
            tgt=x,
            src_mask=causal_mask,  # causal mask (i.e., to prevent us from attending to future tokens in the sequence)
            src_key_padding_mask=torch.where(~attention_mask, 0, -torch.inf)  # masks PAD tokens
        )
        # Compute logits from FC layer: shape (batch_size, seq_len, vocab_size)
        return self.Wout(x_out)  # No softmax as nn.CrossEntropyLoss computes it for us

    def evaluate(
            self,
            input_ids_full_track: torch.Tensor,
            targets_full_track: torch.Tensor,
            mask_full_track: torch.Tensor,
            batch_size: int = 2
    ) -> torch.Tensor:
        """Performs evaluation for a FULL TRACK, returns sum(NLL) / len(raw_sequence_length)"""

        def calculate_nll(inputs_: torch.Tensor, targets_: torch.Tensor, mask_: torch.Tensor) -> torch.Tensor:
            # Through the model
            with torch.no_grad():
                out = self(inputs_, targets_, mask_)
            # Log softmax then compute negative log likelihood
            log_probs = nn.functional.log_softmax(out, dim=-1)  # (batch, seq_length, vocab_size)
            # Gather the log probabilities corresponding to targets
            return -log_probs.gather(dim=-1, index=targets_.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)

        # Remove the batch dimension if we have it
        #  Remember: this function should only be called with all tokens obtained from a SINGLE RECORDING
        if len(input_ids_full_track.size()) == 2:
            input_ids_full_track = input_ids_full_track.squeeze(0)
            targets_full_track = targets_full_track.squeeze(0)
            mask_full_track = mask_full_track.squeeze(0)

        # Decode the sequence to get the non-BPE token sequence
        bpe_decoded = [self.tokenizer.bpe_token_mapping[i.item()] for i in input_ids_full_track]
        bpe_decoded = [x for xs in bpe_decoded for x in xs]  # flatten
        # Compute the length of the decoded sequence: should be equal or larger than the initial sequence
        decoded_seq_len = len(bpe_decoded)
        encoded_seq_len = input_ids_full_track.size(0)
        assert decoded_seq_len >= encoded_seq_len

        # Unfold inputs into (minibatch, max_seq_len)
        targets = targets_full_track.unfold(0, self.max_seq_len, 1).detach()
        inputs = input_ids_full_track.unfold(0, self.max_seq_len, 1).detach()
        masks = mask_full_track.unfold(0, self.max_seq_len, 1).detach()
        # Treat the first full window as a special case
        window1_inputs = inputs[0, :].unsqueeze(0)
        window1_targets = targets[0, :].unsqueeze(0)
        window1_masks = masks[0, :].unsqueeze(0)
        # We want all the NLL values from the first window: subsequent windows, we only want the last NLL value
        all_nlls = calculate_nll(window1_inputs, window1_targets, window1_masks)
        # We need to remove the padding here
        all_nlls = all_nlls[~window1_masks].flatten().tolist()

        # If we need to start sliding the window across to get more than max_seq_len items
        if encoded_seq_len > self.max_seq_len:
            # Split the remaining sliding windows into ((minibatch1, max_seq_len), (minibatch2, max_seq_len), ...)
            inputs_batched = torch.split(inputs[1:, :], batch_size)
            targets_batched = torch.split(targets[1:, :], batch_size)
            masks_batched = torch.split(masks[1:, :], batch_size)
            # Process each (minibatch, max_seq_len) individually
            for window_input, window_target, window_mask in tqdm(
                    zip(inputs_batched, targets_batched, masks_batched),
                    desc=f"Processing track with {encoded_seq_len} encoded tokens, {decoded_seq_len} raw tokens...",
                    total=len(inputs_batched)
            ):
                # Compute the NLL for this batch of sliding windows
                batched_nll = calculate_nll(window_input, window_target, window_mask)
                # We don't have to worry about padding here as we know that the whole input is longer than max_seq_len
                all_nlls.extend(batched_nll[:, -1].flatten().tolist())

        # We should have one NLL value for every input ID in the sequence (after removing padding tokens)
        input_nomask = input_ids_full_track[~mask_full_track]
        target_nomask = targets_full_track[~mask_full_track]
        assert len(all_nlls) == input_nomask.size(0) == target_nomask.size(0)
        # Normalise the sum of all NLL values by the length of the decoded(/raw/non-BPE) sequence and return
        return torch.tensor(sum(all_nlls) / decoded_seq_len)

    @staticmethod
    def _temperature_top_p(
            logits: torch.Tensor,
            top_p: float = DEFAULT_TOP_P,
            temperature: float = DEFAULT_TEMPERATURE,
    ) -> torch.Tensor:
        """Applies temperature and top-p to a tensor of logits (non-normalised)"""
        # Apply temperature scaling
        temp_logits = logits / temperature
        # Compute probabilities
        probs = torch.nn.functional.softmax(temp_logits, dim=-1)
        # Apply top-p (nucleus) filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens outside top-p nucleus
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the removal mask right to always keep at least the top token
        sorted_indices_to_remove = torch.cat(
            [torch.zeros_like(sorted_indices_to_remove[:, :1], dtype=torch.bool),
             sorted_indices_to_remove[:, :-1]],
            dim=-1
        )
        # Scatter the mask back to the original order
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        # Use an out-of-place masked_fill to zero out the unwanted probabilities
        filtered_probs = probs.masked_fill(indices_to_remove, 0.0)
        # Renormalize the probabilities so they sum to 1
        return filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    def sampling(
            self,
            logits: torch.Tensor,
            top_p: float = DEFAULT_TOP_P,
            temperature: float = DEFAULT_TEMPERATURE,
    ):
        """Temperature and nucleus sampling: increases temperature when tokens in nucleus < (vocab_size) / 100"""
        # Apply temperature and top-p to the logits and count how many unique tokens are in the nuclues
        sampled = self._temperature_top_p(logits, top_p=top_p, temperature=temperature)
        in_nucleus = (sampled > 0).sum(dim=-1).item()  # Count per batch
        # We want at least 1% of the vocabulary size to be included in the nucleus
        while in_nucleus < self.tokenizer.vocab_size // 100:
            # Increase temperature by 1% and re-sample
            temperature *= 1.01
            sampled = self._temperature_top_p(logits, top_p=top_p, temperature=temperature)
            in_nucleus = (sampled > 0).sum(dim=-1).item()
        # Return the probability distribution
        return torch.distributions.Categorical(probs=sampled)

    def generate(
            self,
            primer: torch.tensor = None,
            target_seq_length: int = None,
            top_p: float = DEFAULT_TOP_P,
            temperature: float = DEFAULT_TEMPERATURE,
    ):
        """Generates midi given a primer sample with nucleus sampling."""
        assert not self.training, "Cannot generate while in training mode"
        if primer.dim() > 1:
            raise ValueError(f"Expected a tensor with 1 dimension for generation, but got {primer.dim()} dimensions!")
        # Use default max sequence length if not provided
        if target_seq_length is None:
            target_seq_length = self.max_seq_len

        # Create an empty array with the target sequence length
        gen_seq = torch.full(
            (1, target_seq_length),
            self.tokenizer["PAD_None"],
            dtype=torch.long,
            device=utils.DEVICE
        )
        # This counter keeps track of where we are in the sequence
        cur_i = 1
        # Fill in the empty array with our primer sequence if we've provided this
        if primer is not None:
            gen_seq[:, :len(primer)] = primer.type(torch.long).to(utils.DEVICE)
            cur_i = len(primer)
        # Keep iterating until we hit the desired sequence length
        while cur_i < target_seq_length:
            # Unpack everything from the currently generated sequence
            input_ids = gen_seq[:, :cur_i]
            # Predict the next token in the sequence
            next_token, _ = self.predict_next_token(input_ids, top_p, temperature)
            # Add the next token into the sequence
            gen_seq[:, cur_i] = next_token
            # Increment the counter
            cur_i += 1
            # Let the transformer decide to end the sequence early if it wants to
            if next_token == self.tokenizer["EOS_None"] or cur_i >= target_seq_length:
                break
        return gen_seq[:, :cur_i]

    def predict_next_token(
            self,
            inputs: torch.Tensor,
            top_p: float = DEFAULT_TOP_P,
            temperature: float = DEFAULT_TEMPERATURE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given the current inputs, sample the next token: returns the token itself + associated log probabilities"""
        attention_mask = create_padding_mask(inputs, self.tokenizer.pad_token_id)
        # Through the model to get the logits: shape (1, sequence_length, vocab_size)
        with torch.no_grad():
            logits = self(inputs, None, attention_mask)
        # This gets the probabilities for the next token: shape (1, vocab_size)
        token_probs = logits[:, -1, :]  # no need to softmax, we do this in _temperature_top_p
        # Get the next token after applying top-p + temperature sampling
        m = self.sampling(token_probs, top_p, temperature)
        action = m.sample()
        # Return the predicted next token and the log probabilities
        return action, m.log_prob(action)


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
    import argparse

    from loguru import logger

    from jazz_style_conditioned_generation.data.tokenizer import (
        load_tokenizer,
        add_genres_to_vocab,
        add_pianists_to_vocab,
        add_tempos_to_vocab,
        add_timesignatures_to_vocab,
        add_recording_years_to_vocab
    )
    from jazz_style_conditioned_generation.data.dataloader import DatasetMIDIConditionedRandomChunk
    from jazz_style_conditioned_generation.metrics import cross_entropy_loss

    utils.seed_everything(utils.SEED)
    n_midis = 20

    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Test out model configuration by forwards/backwards passing one batch")
    parser.add_argument("-b", "--batch-size", type=int, help="Batch size to use", default=6)
    parser.add_argument("-s", "--max-seq-len", type=int, help="Maximum sequence length", default=1024)
    parser.add_argument("-m", "--dim-model", type=int, help="Model hidden dimension", default=768)
    parser.add_argument("-f", "--dim-feedforward", type=int, help="Feed hidden dimension", default=2048)
    parser.add_argument("-e", "--num-heads", type=int, help="Number of attention heads", default=8)
    parser.add_argument("-l", "--num-layers", type=int, help="Number of transformer layers", default=12)
    parser.add_argument("-d", "--dropout", type=float, help="Dropout fraction on feedforward layer", default=0.1)
    # Parse all arguments from the command line
    parser_args = vars(parser.parse_args())
    logger.info(f"Creating music transformer with kwargs: {parser_args}", )
    # Get a small number of MIDI files
    midis = utils.get_data_files_with_ext("data/pretraining", "**/*.mid")
    midis = midis[:n_midis]
    # Create and train the tokenizer with the given vocabulary size
    toker = load_tokenizer(tokenizer_str="tsd")
    add_tempos_to_vocab(toker, 80)
    add_timesignatures_to_vocab(toker, [3, 4])
    add_pianists_to_vocab(toker)
    add_genres_to_vocab(toker)
    add_recording_years_to_vocab(toker)
    # Create the dataset that returns full-length tracks
    ds = torch.utils.data.DataLoader(
        DatasetMIDIConditionedRandomChunk(
            tokenizer=toker,
            files_paths=midis,
            do_conditioning=False,  # no conditioning for now
            do_augmentation=False,
            max_seq_len=parser_args["max_seq_len"],
        ),
        batch_size=parser_args["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    # Create the model and set to training mode
    try:
        mt = MusicTransformer(
            tokenizer=toker,
            max_seq_len=parser_args["max_seq_len"],
            d_model=parser_args["dim_model"],
            dim_feedforward=parser_args["dim_feedforward"],
            num_heads=parser_args["num_heads"],
            dropout=parser_args["dropout"],
            n_layers=parser_args["num_layers"],
        ).to(utils.DEVICE)
        # Create the optimizer
        opt = torch.optim.Adam(mt.parameters(), lr=1e-4, )
        # Get a single batch
        batch = next(iter(ds))
        iids = batch["input_ids"].to(utils.DEVICE)
        targets = batch["labels"].to(utils.DEVICE)
        ctoks = batch["condition_ids"].to(utils.DEVICE)
        amask = batch["attention_mask"].to(utils.DEVICE)
        # Forwards pass
        out = mt(iids, targets, amask, ctoks, )
        loss = cross_entropy_loss(out, targets, toker)
        # Backwards pass
        opt.zero_grad()
        loss.backward()
        opt.step()
    except (torch.OutOfMemoryError, RuntimeError) as e:
        logger.warning(f"... oof, getting OOMs with those settings! error: {e}")
    else:
        logger.info("... completed backwards pass of a single batch with no OOMs, good to go with those settings!")
