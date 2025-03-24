#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Find ideal LR using the training module. Pretty similar to pytorch lightning method."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from jazz_style_conditioned_generation import utils, training


class LRTrainingModule(training.TrainingModule):
    """Overrides base training module to find LR"""

    num_steps = 1000
    init_lr = 1e-8
    end_lr = 1.0
    beta = 0.98  # smoothing value: controls the forget rate
    gamma = (end_lr / init_lr) ** (1 / num_steps)  # exponential increasing every step
    early_stop_thresh = 4.0  # end experiment early if current_loss > best_loss * thresh

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.train()  # need to be in training mode here
        # Reinitialise the optimizer with the desired learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)
        self.best_step = None
        self.best_lr = None
        self.best_validation_loss = float('inf')
        self.average_validation_loss = 0.
        # We don't care about the scheduler here
        self.res = []
        self.df = None
        self.optimal_idx = None

    def start(self):
        """Runs LR finding experiment, overriding base training method"""

        # Convert the dataloader to an iterator, makes it easy to do a smaller number of steps
        train_data_iter = iter(self.train_loader)
        lr = self.init_lr

        # Make the desired number of steps
        for step in tqdm(range(self.num_steps + 1), total=self.num_steps, desc=f"Doing {self.num_steps} steps..."):
            # Get the next batch from the dataloader
            try:
                batch = next(train_data_iter)
            # Reset dataloader if exhausted
            except StopIteration:
                train_data_iter = iter(self.train_loader)
                batch = next(train_data_iter)

            # Set the learning rate for this batch across all parameter groups
            for param_grp in self.optimizer.param_groups:
                param_grp["lr"] = lr

            # Forwards pass through the model to calculate loss
            loss, _ = self.step(batch)

            # Backwards pass with the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Avg loss (loss with momentum) + smoothing
            self.average_validation_loss = self.beta * self.average_validation_loss + (1 - self.beta) * loss.item()
            loss = self.average_validation_loss / (1 - self.beta ** (step + 1))

            # Update attributes if required
            if loss < self.best_validation_loss:
                self.best_validation_loss = loss
                self.best_step = step
                self.best_lr = lr

            # Break when loss exceeds best_loss * thresh
            if loss >= self.best_validation_loss * self.early_stop_thresh:
                logger.info(f"Early stopping at step {step}, loss = {loss:.3f}")
                break

            # Log every 100 steps
            if step % 100 == 0:
                logger.info(f"Loss at step {step}, lr {lr} is {loss:.3f}")

            # Append the loss and learning rate
            self.res.append(dict(
                step=step,
                loss=loss,
                lr=lr
            ))

            # Increase learning rate accordingly
            lr *= self.gamma

        # Dump results to a CSV
        logger.info(f"Done: best loss {self.best_validation_loss:.3f}, step {self.best_step}, lr {self.best_lr}")
        self.df = pd.DataFrame(self.res)
        self.df.to_csv(os.path.join(utils.get_project_root(), f"references/lr_experiment_{self.run}.csv"))
        logger.info(f"Suggested learning rate is {self.suggestion()}")

    def suggestion(self, skip_begin: int = 10, skip_end: int = 1) -> float:
        """See equivalent method in pytorch_lightning.tuner.lr_finder"""

        # Create arrays and omit the skip indexes: prevents too optimistic estimates
        loss_arr = np.array([i["loss"] for i in self.res])[skip_begin:-skip_end]
        lr_arr = np.array([i["lr"] for i in self.res])[skip_begin:-skip_end]
        # Subset, compute gradient
        loss = loss_arr[np.isfinite(loss_arr)]
        min_grad = np.gradient(loss).argmin()
        self.optimal_idx = min_grad + skip_begin
        # Return suggested learning rate
        return lr_arr[self.optimal_idx]


if __name__ == "__main__":
    utils.seed_everything(utils.SEED)

    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Plot a learning rate graph for the given model")
    parser.add_argument("-c", "--config", type=str, help="Path to config YAML file")
    # Parse all arguments from the provided YAML file
    parser_args = vars(parser.parse_args())
    if not parser_args:
        raise ValueError("No config file specified")
    training_kws = training.parse_config_yaml(parser_args['config'])

    # We need to set a few arguments to false always
    training_kws["_generate_only"] = False  # should only be set to True when running generate.py
    training_kws["mlflow_cfg"]["use"] = False
    training_kws["checkpoint_cfg"]["save_checkpoints"] = False
    training_kws["checkpoint_cfg"]["load_checkpoints"] = False
    training_kws["batch_size"] = 2  # should help with larger models

    # Create the training module
    lrt = LRTrainingModule(**training_kws)
    # Start finding the LR
    lrt.start()
