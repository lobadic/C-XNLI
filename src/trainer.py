import os
from loguru import logger
from typing import Optional
from packaging import version

import json
from pathlib import Path

import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import random
import numpy as np
from tqdm import tqdm, trange


from transformers import get_scheduler

from src.optimizer import get_optimizer
from src.utils import multilevel_dict_lookup


_PREFIX_CHECKPOINT_DIR = "checkpoint"


class Trainer:
    def __init__(
        self,
        args,
        model,
        evaluator,
        optimizer_name,
        optimizer_kwargs,
        lr_scheduler_name,
        train_dataset,
        eval_dataset,
        output_dir: str,
        device: torch.device,
        evaluate_during_training: bool,
        num_train_epochs: int,
        train_batch_size: int,
        eval_batch_size: int,
        warmup_steps: int,
        warmup_proportion: float,
        gradient_accumulation_steps: int,
        logging_steps: int,
        max_grad_norm: float,
        logging_first_step: bool = True,
        num_workers: int = 8,
        save_best_metric: list = ["cls_report", "weighted avg", "f1-score"],
        save_best_only: bool = False,
        save_best_mode: str = "max",
        save_best_patience: Optional[int] = None,
        best_metric_minimal_gain: float = 0.0,
        save_model_checkpoints: bool = True,
        **kwargs,
    ):

        self.args = args
        self.model = model
        self.evaluator = evaluator
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_name = lr_scheduler_name
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.evaluate_during_training = evaluate_during_training
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.warmup_steps = warmup_steps
        self.warmup_proportion = warmup_proportion
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_best_patience = save_best_patience
        self.logging_steps = logging_steps
        self.logging_first_step = logging_first_step
        self.max_grad_norm = max_grad_norm
        self.num_workers = num_workers
        self.save_best_only = save_best_only
        self.save_best_mode = save_best_mode
        assert self.save_best_mode in ["max", "min"]
        self.save_best_metric = save_best_metric
        self.best_metric_minimal_gain = best_metric_minimal_gain
        assert self.best_metric_minimal_gain >= 0.0
        self.save_model_checkpoints = save_model_checkpoints

        # to be initialized: (all in one place for readability)
        self.update_steps_per_epoch = None
        self.t_total = None
        self.global_step = None
        self.epoch = None
        self.total_train_batch_size = None
        self.best_metric = (
            -float("inf") if self.save_best_mode == "max" else float("inf")
        )

        # contains args, params, metrics for hyperparam optimization
        self.best_metadata = None

        self.device = device
        logger.info(f"Using device '{self.device}'")

        if self.save_model_checkpoints:
            if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
                raise ValueError(
                    f"Output directory ({self.output_dir}) already exists and is not empty."
                )
            else:
                Path(self.output_dir).mkdir(parents=True)

        self.model = self.model.to(self.device)

        self.train_dataloader = self._get_dataloader(ds_subset="train")
        self.eval_dataloader = self._get_dataloader(ds_subset="eval")

        self.num_batches_per_epoch = len(self.train_dataloader)
        self.update_steps_per_epoch = (
            self.num_batches_per_epoch // self.gradient_accumulation_steps
        )
        self.t_total = self.update_steps_per_epoch * self.num_train_epochs
        logger.info(f"Total number of update steps: {self.t_total}")

        if self.warmup_proportion:
            self.warmup_steps = int(self.t_total * self.warmup_proportion)
            logger.info(
                f"Warmup proportion: {self.warmup_proportion}, Warmup steps: {self.warmup_steps}"
            )
        else:
            logger.info(f"Warmup update steps: {self.warmup_steps}")

        self.optimizer = get_optimizer(
            self.optimizer_name, model=self.model, **self.optimizer_kwargs
        )
        self.lr_scheduler = get_scheduler(
            self.lr_scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.t_total,
        )

        self.total_train_batch_size = (
            self.train_batch_size * self.gradient_accumulation_steps
        )

    def _get_output_dir(
        self,
    ):
        return os.path.join(
            self.output_dir, f"{_PREFIX_CHECKPOINT_DIR}-{self.global_step}"
        )

    def _training_step(self, model, batch) -> float:

        model.train()

        if hasattr(model, "prepare_inputs") and callable(model.prepare_inputs):
            inputs = model.prepare_inputs(batch, device=self.device)
        else:
            inputs = {k: v.to(self.device) for k, v in batch.items()}

        loss, outputs = model(**inputs, return_dict=False)

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        loss.backward()

        return loss.item()

    def _clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def _step(self):
        self.optimizer.step()
        self.lr_scheduler.step()

    def _get_train_sampler(self) -> torch.utils.data.sampler.Sampler:
        return RandomSampler(self.train_dataset)

    def _get_eval_sampler(self) -> torch.utils.data.sampler.Sampler:
        return SequentialSampler(self.eval_dataset)

    def update_best_metric(self, current_metric):

        to_update: bool = False

        if self.save_best_mode == "max":
            if current_metric > self.best_metric * (1 + self.best_metric_minimal_gain):
                to_update = True

        elif self.save_best_mode == "min":
            if current_metric < self.best_metric * (1 - self.best_metric_minimal_gain):
                to_update = True

        if to_update:
            logger.info(
                f"Updating best metric value from {self.best_metric} to {current_metric}"
            )
            self.best_metric = current_metric

        return to_update

    def _get_dataloader(
        self,
        ds_subset: str,
        test_dataset=None,
        test_batch_size: Optional[int] = None,
    ):

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if ds_subset == "train":
            dataset = self.train_dataset
            sampler = self._get_train_sampler()
            batch_size = self.train_batch_size
        elif ds_subset == "eval":
            dataset = test_dataset if test_dataset is not None else self.eval_dataset
            sampler = self._get_eval_sampler()
            batch_size = test_batch_size if test_batch_size else self.eval_batch_size
        else:
            raise ValueError("Wrong dataset subset. Only 'train' and 'eval' supported.")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def train(self, model_dir: Optional[str] = None):

        logger.info("***** Running training *****")
        logger.info(f"Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"Num batches = {self.num_batches_per_epoch}")
        logger.info(f"Num Epochs = {self.num_train_epochs}")
        logger.info(
            f"Training batch size {self.train_batch_size}",
        )
        logger.info(f"Total train batch size = {self.total_train_batch_size}")
        logger.info(f"Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"Total optimization steps = {self.t_total}")

        self.global_step = 0
        self.epoch = 0.0

        self.tr_loss = 0.0

        self.model.train()
        self.model.zero_grad()

        patience_counter = 0

        train_iterator = trange(
            0,
            self.num_train_epochs,
            desc="Epoch",
        )
        for epoch in train_iterator:

            epoch_iterator = tqdm(
                self.train_dataloader,
                desc="Iteration",
            )
            for step, batch in enumerate(epoch_iterator):

                self.tr_loss += self._training_step(self.model, batch)

                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    self._clip_gradients()
                    self._step()
                    self.model.zero_grad()

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    epoch_iterator.set_description(
                        "Avg loss: {:.9f}".format(self.tr_loss / self.global_step)
                    )

                    # LOGGING
                    if (
                        self.logging_steps > 0
                        and self.global_step % self.logging_steps == 0
                    ) or (self.global_step == 1 and self.logging_first_step):
                        self._log()

            # ON EPOCH END:

            # LOG
            self._log()

            # EVALUATE
            if self.evaluate_during_training:
                metrics = self._evaluate()
                logger.info(metrics)

                for k, v in metrics.items():
                    if "print" in k:
                        print(k)
                        print(v)

            if self.save_best_only and self.evaluate_during_training:
                current_metric = multilevel_dict_lookup(metrics, self.save_best_metric)
                current_is_best = self.update_best_metric(current_metric)
            else:
                current_is_best = False

            if current_is_best:
                patience_counter = 0

                self.best_metadata = {
                    "args": vars(self.args),
                    "metrics": metrics,
                    "other_params": {
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "training_loss": self.tr_loss,
                    },
                }
            else:
                patience_counter += 1

            if (self.save_best_patience is not None) and (
                patience_counter >= self.save_best_patience
            ):
                logger.info("Patience surpassed, stopping training.")
                break

            # SAVE
            if self.save_model_checkpoints:
                if current_is_best or not self.save_best_only:
                    output_dir = self._get_output_dir()
                    self.save_model(output_dir)

        logger.info("\n\nTraining completed.")
        return self.best_metadata

    def _log(
        self,
    ):
        logs = {}

        logs["global_step"] = self.global_step
        logs["epoch"] = self.epoch
        logs["average_loss"] = self.tr_loss / self.global_step

        logs["learning_rate"] = (
            self.lr_scheduler.get_last_lr()[0]
            if version.parse(torch.__version__) >= version.parse("1.4")
            else self.lr_scheduler.get_lr()[0]
        )

        logger.info(logs)

    def _write_evaluation_results(self, outpath, logs):

        if Path(outpath).exists():
            out_mode = "a"
        else:
            out_mode = "w"

        # writing logs
        with open(outpath, out_mode) as writer:
            writer.write(
                f"\nEvaluating on the {os.path.abspath(self.args.dev_filepath)} dataset.\n"
            )
            writer.write(f"Global step = {self.global_step}\n")
            writer.write(f"Epoch = {self.epoch}\n")
            for key in logs.keys():
                if "confusion_matrix" in key:
                    writer.write("%s = \n%s\n" % (key, str(logs[key])))
                else:
                    writer.write("%s = %s\n" % (key, str(logs[key])))

    def _evaluate(self, **kwargs):

        self.model.eval()
        eval_dataloader = self._get_dataloader(ds_subset="eval")
        self.evaluator.evaluation_loop(
            model=self.model,
            dataloader=eval_dataloader,
            num_labels=self.args.num_outputs,
            device=self.device,
        )
        metrics = self.evaluator.get_metrics(**kwargs)

        if self.save_model_checkpoints:
            output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
            self._write_evaluation_results(output_eval_file, metrics)

        return metrics

    def _save_training_args(self, output_dir, indent=4, sort_keys=True):

        args = self.args

        logger.info(
            f"Saving training args to {output_dir}",
        )
        with open(os.path.join(output_dir, "training_args.json"), "w") as outfile:
            args_dict = vars(args)
            args_dict["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            json.dump(args_dict, outfile, indent=indent, sort_keys=sort_keys)

    def save_model(
        self,
        output_dir,
        save_optimizer: bool = True,
    ):
        output_dir = output_dir if output_dir else self.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Saving model checkpoint to {os.path.abspath(output_dir)}",
        )

        if hasattr(self.model, "save_model") and callable(self.model.save_model):
            self.model.save_model(output_dir)
        else:
            self.model.save_pretrained(output_dir)

        if hasattr(self.train_dataset, "tokenizer"):
            self.train_dataset.tokenizer.save_pretrained(output_dir)

        if save_optimizer:
            logger.info(
                f"Saving optimizer and lr scheduler to {output_dir}",
            )
            torch.save(
                self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
            )
            torch.save(
                self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
            )
        # Good practice: save your training arguments together with the trained model
        self._save_training_args(output_dir)
