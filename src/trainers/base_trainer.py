from typing import Optional

import torch

import os
from tqdm import tqdm

import wandb
import huggingface_hub as hf

from omegaconf import DictConfig, OmegaConf

from utils.import_utils import import_optimizer, import_collator
import utils.constants as constants
from utils.dot_dict import DotDict
from utils.logging_utils import LogSection


class BaseTrainer:

    def __init__(
        self,
        config: DictConfig,
        model,
        dataset,
    ):
        self.config = config
        self.model = model
        self.dataset = dataset

        self.checkpoint_path = None
        # self.repo_name = None
        if not self.config.debug:

            os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)

            # create the huggingface save repo
            full_name = f"{self.config.project}_{self.config.name}"
            
            self.checkpoint_path = os.path.join(
                constants.LOCAL_DATA_PATH,
                full_name
            )
            # self.repo_name = f"{constants.HF_ID}/{full_name}"

            # hf.create_repo(
            #     self.repo_name, private=True, exist_ok=True, token=constants.HF_TOKEN
            # )

            # create the wandb project
            wandb.init(
                project=self.config.project,
                name=self.config.name,
                notes=self.config.notes,
                config=OmegaConf.to_container(self.config, resolve=True),
            )


    def safe_log(self, **kwargs):
        if self.config.debug:
            return

        d = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.item()
            else:
                d[k] = v

        wandb.log(d)
        

    @torch.no_grad()
    def save_checkpoint(
        self,
        model,
        step
    ):
        if self.config.debug:
            return

        with LogSection(f"step {step} checkpoint saving"):

            path = os.path.join(
                self.checkpoint_path,
                f"step_{step:012d}"
            )
            model.save_pretrained(
                path,
                push_to_hub=False,
            )


    def train(self):

        # init model
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.train()
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # init optimizer
        optimizer = import_optimizer(self.config.optimizer.type)(
           self.model.parameters(),
            **self.config.optimizer.kwargs
        )

        # init data loader
        collator = import_collator(self.config.collator.type)(
            **self.config.collator.kwargs
        )
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=collator,
            shuffle=False,
            drop_last=True,
        )

        # init loop vars
        step = 0
        pbar = tqdm(desc=f"Training {self.project}/{self.name}")
        pbar.update(0)

        # train indefinitely
        while True:
            for batch in loader:

                loss, aux = self.train_step(
                    step,
                    optimizer,
                    **batch
                )

                # update tracking
                step += 1

                # update pbar
                pbar.update(1)
                pbar.set_postfix(
                    loss=loss.item() if isinstance(loss, torch.Tensor) else loss
                )

                # log to wandb
                self.safe_log(loss=loss, **aux)

                # save checkpoint
                if step % self.checkpoint_interval == 0:
                    self.save_checkpoint(step)

        # except KeyboardInterrupt:

        #     if not self.config.debug:
        #         self.optimizer.zero_grad(set_to_none=True)

        #         print("Training interrupted! Saving checkpoint...")
        #         self.save_checkpoint(step)

        #     raise KeyboardInterrupt("Training interrupted by user.")
    

    def train_step(
        self,
        step,
        optimizer,
        **batch
    ):
        
        with torch.autocast(
            device_type=str(constants.DEVICE),
            dtype=self.config.autocast_dtype,
        ):
            loss, aux = self.train_forward(
                step,
                **batch
            )
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        return loss, aux


    def train_forward(
        self,
        step,
        **batch
    ):
        raise NotImplementedError("train_forward must be implemented in child class!")
    

    def debug_gradients(self):

        with open(os.path.join(constants.LOCAL_DATA_PATH, "gradients.txt"), "w") as f:

            f.write("\n === FINITE GRADIENT === \n\n")
            for n, p in self.model.named_parameters():
                if p.grad is not None and torch.all(p.grad.isfinite()):
                    f.write(f"{n}\n")

            f.write("\n === NO GRADIENT === \n\n")
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    f.write(f"{n}\n")

            f.write("\n === NONFINITE GRADIENT === \n\n")
            for n, p in self.model.named_parameters():
                if p.grad is not None and torch.any(~p.grad.isfinite()):
                    f.write(f"{n}\n")
