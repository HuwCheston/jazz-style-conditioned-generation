#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation of cross-entropy loss that decodes BPE-encoded tokens to the base vocabulary"""

import numpy as np
import torch
import torch.nn.functional as F
from miditok import MIDILike, MusicTokenizer


def cross_entropy_loss(
        batched_logits: torch.Tensor,
        batched_labels: torch.Tensor,
        tokenizer: MusicTokenizer
) -> torch.Tensor:
    """Cross-entropy loss hack that decodes BPE-encoded tokens"""
    ys, ts = [], []
    # This iterates through every item: logits (bpe_sequence_len, bpe_vocab), labels (bpe_sequence_len)
    for logits, labels in zip(batched_logits, batched_labels):
        all_ys, all_ts = [], []
        # Iterating through (BPE-tokenized) steps in the sequence
        for log, lab in zip(logits, labels):
            # Decode target to a list of base tokens
            lab_decoded = tokenizer.bpe_token_mapping[lab.item()]
            all_ts.extend(lab_decoded)
            # Create a mapping of [decoded_step_idx1: [[base_token1, ...], [base_token2, ...], [base_token3, ...]]]
            results_at_step = [[[] for __ in range(tokenizer.base_vocab_size)] for _ in range(len(lab_decoded))]
            # Iterate over BPE token IDX, non-normalised probability at this BPE step
            for log_idx, log_prob in enumerate(log.tolist()):
                # Decode BPE token IDX to a list of base tokens
                log_idx_decoded = tokenizer.bpe_token_mapping[log_idx]
                # The length of both lists might be different
                #  If we have more target labels than predicted labels
                if len(log_idx_decoded) < len(lab_decoded):
                    # Pad the sequence by continuing to predict the final token
                    overlap = len(lab_decoded) - len(log_idx_decoded)
                    log_idx_decoded = log_idx_decoded + [log_idx_decoded[-1] for _ in range(overlap)]
                #  If we have fewer target labels than predicted labels
                elif len(log_idx_decoded) > len(lab_decoded):
                    # Truncate the predicted labels to match the length of the predicted labels
                    log_idx_decoded = log_idx_decoded[:len(lab_decoded)]
                # Lengths should now match
                # Smooth the probability over all the decoded tokens
                #  So, if we decode to 2 base tokens with a probability of 1.
                #  We assign a probability of 0.5 to each decoded base token
                smoothed_log_prob = log_prob / len(log_idx_decoded)
                for step_idx, decoded_token in enumerate(log_idx_decoded):
                    results_at_step[step_idx][decoded_token].append(smoothed_log_prob)
            # Sum everything so we get a single probability for each base token at every step
            all_ys.extend([[sum(v) for v in r] for r in results_at_step])
        # (decoded_sequence_len, base_vocab_size)
        ys.append(torch.tensor(all_ys))
        # (decoded_sequence_len,)
        ts.append(torch.tensor(all_ts))

    # Pad all the tensors to the length of the longest decoded sequence
    max_len = max(i.size(0) for i in ts)
    ys_padded = torch.stack([F.pad(y, pad=(0, 0, 0, max_len - y.size(0)), value=tokenizer.pad_token_id) for y in ys])
    ts_padded = torch.stack([F.pad(s, pad=(0, max_len - s.size(0)), value=tokenizer.pad_token_id) for s in ts])
    # Now, truncate to the length of the originally BPE-encoded sequence
    ys_trunc = ys_padded[:, :batched_labels.size(1), :]  # (batch_size, seq_len, vocab_size)
    ts_trunc = ts_padded[:, :batched_labels.size(1)]  # (batch_size, seq_len)
    # Everything else can just use the torch function
    return vanilla_cross_entropy_loss(ys_trunc, ts_trunc, tokenizer)


def vanilla_cross_entropy_loss(
        batched_logits: torch.Tensor,
        batched_labels: torch.Tensor,
        tokenizer: MusicTokenizer
):
    """Just implements the vanilla cross entropy loss from torch, with some reshaping"""
    return F.cross_entropy(
        batched_logits.reshape(batched_logits.shape[0] * batched_logits.shape[1], -1).to(torch.float),
        batched_labels.flatten().to(torch.long),
        ignore_index=tokenizer.pad_token_id
    )


# Set some random seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# First, use a tokenizer that hasn't been trained
t = MIDILike()
token_mapping = {
    0: [0],  # This just maps a base token ID onto a list of itself
    1: [1],  # Might seem redundant, but it allows us to use the same function with a trained/non-trained tokenizer
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6]
}
# Set everything as attributes of the tokenizer
setattr(t, "bpe_token_mapping", token_mapping)
setattr(t, "base_vocab_size", 7)  # the number of "unique" items in all the values of token_mapping
# (batch_size, seq_length)
print("Creating random logits and targets...")
labs = torch.tensor([
    [1, 2, 3, 4, 5, 6, 0, 0, 0, 0],
    [3, 2, 3, 5, 6, 6, 6, 0, 0, 0],
])
# (batch_size, seq_length, vocab_size)
logs = torch.rand((labs.size(0), labs.size(1), len(token_mapping)))
# The loss calculated with our function should be identical to the vanilla torch cross entropy loss
our_loss = cross_entropy_loss(logs, labs, t).item()
vanilla_loss = vanilla_cross_entropy_loss(logs, labs, t).item()
assert our_loss == vanilla_loss
print(f"Without BPE, our loss is: {our_loss:.3f}")
print(f"Without BPE, vanilla torch loss is: {vanilla_loss:.3f}")

# Second, "train" the tokenizer (actually, just use a hack to simulate a trained tokenizer)
t = MIDILike()
print("Creating random logits and targets...")
token_mapping = {
    0: [0],  # Now, this maps a BPE token IDX onto a list of base token IDXs
    1: [1],  # The values are the IDXs of our "base" vocabulary
    2: [2],  # And the keys are the IDXs of the tokens learned with BPE
    3: [3],  # This means that some BPE token IDXs can map onto MULTIPLE base token IDXs!
    4: [1, 2],
    5: [1, 3],
    6: [1, 2, 3]
}
setattr(t, "bpe_token_mapping", token_mapping)  # hack, will be set in train_tokenizer
setattr(t, "base_vocab_size", 4)  # the number of "unique" items in all the values of token_mapping
# (batch_size, bpe_seq_length)
labs = torch.tensor([
    [1, 2, 3, 4, 5, 6, 0, 0, 0, 0],  # decodes to [1, 2, 3, 1, 2, 1, 3, 1, 2, 3, 0, 0, 0, 0]
    [3, 2, 3, 5, 6, 6, 6, 0, 0, 0]  # decodes to [3, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0]
])
# (batch_size, bpe_seq_length, bpe_vocab_size)
logs = torch.rand((labs.size(0), labs.size(1), len(token_mapping)))
# The loss calculated with our function should be smaller than the vanilla torch cross entropy loss
our_loss = cross_entropy_loss(logs, labs, t)
vanilla_loss = vanilla_cross_entropy_loss(logs, labs, t)
assert our_loss.item() < vanilla_loss.item()
print(f"With BPE, our loss is: {our_loss:.3f}")
print(f"With BPE, vanilla torch loss is: {vanilla_loss:.3f}")
