#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom LR scheduler, taken from https://github.com/gwinndr/MusicTransformer-Pytorch"""

import math


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
