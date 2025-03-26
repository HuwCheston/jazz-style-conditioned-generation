#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom LR scheduler classes"""

import math

import torch


class MusicTransformerScheduler:
    """Class for custom learn rate scheduler (to be used by torch.optim.lr_scheduler.LambdaLR).

    Learn rate for each step (batch) given the warmup steps is:
        lr = [ 1/sqrt(d_model) ] * min[ 1/sqrt(step) , step * (warmup_steps)^-1.5 ]
    """

    def __init__(self, model_dim=512, warmup_steps=4000, init_steps=0):
        # Store Values
        self.warmup_steps = warmup_steps
        self.model_dim = model_dim
        self.init_steps = init_steps
        # Begin Calculations
        self.invsqrt_dim = (1 / math.sqrt(model_dim))
        self.invsqrt_warmup = (1 / (warmup_steps * math.sqrt(warmup_steps)))

    def step(self, step):
        """Method to pass to LambdaLR. Increments the step and computes the new learn rate."""
        step += self.init_steps
        if step <= self.warmup_steps:
            return self.invsqrt_dim * self.invsqrt_warmup * step
        else:
            invsqrt_step = (1 / math.sqrt(step))
            return self.invsqrt_dim * invsqrt_step


class WarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Warmup scheduler that increases linearly to max_lr over warmup_steps, then decreases expoenentially to min_lr"""

    def __init__(self, optimizer, max_lr: float, warmup_steps: int, gamma: float):
        self.min_lr = optimizer.param_groups[0]['lr']  # treat the optimizer learning rate as our minimum
        self.max_lr = max_lr
        assert max_lr > self.min_lr, f"`lr` must be smaller than `max_lr`, but got {self.min_lr} and {self.max_lr}"
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super(WarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        # We're in the warmup phase
        if self.last_epoch < self.warmup_steps:
            mult = self.min_lr + (self.max_lr - self.min_lr) * (self.last_epoch / self.warmup_steps)
        # We're inbetween the warmup and decay phases
        elif self.last_epoch == self.warmup_steps:
            mult = self.max_lr
        # We're in the decay phase
        else:
            # This clamps to the minimum desired learning rate
            mult = max(self.min_lr, self.max_lr * (self.gamma ** (self.last_epoch - self.warmup_steps)))
        return [mult for _ in self.optimizer.param_groups]


class DummyScheduler(torch.optim.lr_scheduler.LRScheduler):
    """An LR scheduler that does not modify anything but has the same API as all other schedulers."""

    def __init__(self, optimizer, last_epoch=-1):
        super(DummyScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Just returns the current learning rates without modification"""
        return [group['lr'] for group in self.optimizer.param_groups]
