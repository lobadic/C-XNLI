import os
import argparse
import json
from loguru import logger

import torch

from src.data import ClassificationDataset
from src.evaluators import get_evaluator
from src.trainer import Trainer
from src.models import get_model
from src.utils import set_seed


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", required=True, type=str, help=".")

    # DATA ARGS:
    parser.add_argument(
        "--train_filepath", required=True, type=str, help="Path to training dataset."
    )
    parser.add_argument(
        "--dev_filepath", required=True, type=str, help="Path to dev dataset."
    )
    parser.add_argument(
        "--sentence1_colname",
        required=True,
        type=str,
        help="Name of the column containing first sentence.",
    )
    parser.add_argument(
        "--sentence2_colname",
        required=True,
        type=str,
        help="Name of the column containing second sentence.",
    )
    parser.add_argument(
        "--label_colname",
        required=True,
        type=str,
        help="Name of the column containing labels.",
    )
    parser.add_argument(
        "--possible_labels", type=str, nargs="+", default=[], help="Possible labels."
    )
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        choices=["classification"],
        help="Training task.",
    )

    parser.add_argument(
        "--tokenizer_name",
        required=False,
        default="",
        type=str,
        help="Name of the tokenizer.",
    )
    parser.add_argument(
        "--max_seq_length",
        required=False,
        default=128,
        type=int,
        help="Name of the tokenizer.",
    )

    # MODEL ARGS:
    parser.add_argument(
        "--model_type",
        type=str,
        default="",
        help="Model type.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
        help="Path to the model or pretrained model name.",
    )
    parser.add_argument(
        "--model_init",
        action="store_true",
        help="Whether to train model from scratch.",
    )
    parser.add_argument(
        "--num_outputs",
        type=int,
        required=True,
        help="Number of model outputs.",
    )

    # OPTIMIZER ARGS:
    parser.add_argument(
        "--optimizer_name",
        type=str,
        required=True,
        help="Optimizer name.",
    )
    parser.add_argument(
        "--optimizer_kwargs",
        type=json.loads,
        default="{}",
        help="Optimizer other kwargs",
    )

    # SCHEDULER ARGS:
    parser.add_argument(
        "--lr_scheduler_name",
        type=str,
        required=True,
        help="Lr scheduler name.",
    )

    # TRAINER KWARGS:
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output dir where model ckpt file will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Training seed.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        required=True,
        help="Training batch size.",
    )
    parser.add_argument(
        "--dev_batch_size",
        type=int,
        required=True,
        help="Dev batch size.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Num warmup steps.",
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.0,
        help="Percent of warmup steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        required=True,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=-1,
        help="Num logging steps.",
    )
    parser.add_argument(
        "--save_best_metric",
        required=False,
        nargs="+",
        default=["cls_report", "weighted avg", "f1-score"],
        type=str,
        help="Metric which will be used to determine best model. Supports nesting since metrics object can be nested dict",
    )
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        help="Whether to save only best models.",
    )
    parser.add_argument(
        "--save_best_mode",
        type=str,
        default="max",
        help="Save best mode: 'max' or 'min'.",
    )
    parser.add_argument(
        "--save_best_patience",
        type=int,
        default=None,
        help="Patience.",
    )
    parser.add_argument(
        "--best_metric_minimal_gain",
        type=float,
        default=0.0,
        help="Minimal percentage of `save_best_metric` change needed in order to set the model as the current best model during training. Note: set 5 %/ as 0.05",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm. Everything above will be clipped.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )

    return parser.parse_args()


def main():

    args = get_args()
    logger.info(f"Arguments: {args}")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    set_seed(seed=args.seed)

    config, tokenizer, model = get_model(
        model_type=args.model_type,
        model_init=args.model_init,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name=args.tokenizer_name,
        num_outputs=args.num_outputs,
    )
    model.to(device)

    train_dataset = ClassificationDataset(
        filepath=args.train_filepath,
        text_columns=[args.sentence1_colname, args.sentence2_colname],
        label_column=args.label_colname,
        possible_labels=args.possible_labels,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    dev_dataset = ClassificationDataset(
        filepath=args.dev_filepath,
        text_columns=[args.sentence1_colname, args.sentence2_colname],
        label_column=args.label_colname,
        possible_labels=args.possible_labels,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    if args.task == "classification":
        assert args.num_outputs == len(
            args.possible_labels
        ), f"Number of model outputs ({args.num_outputs}) does not match number of possible labels ({len(args.possible_labels)})."

    evaluator = get_evaluator(task=args.task)

    trainer = Trainer(
        args=args,
        model=model,
        evaluator=evaluator,
        optimizer_name=args.optimizer_name,
        optimizer_kwargs=args.optimizer_kwargs,
        lr_scheduler_name=args.lr_scheduler_name,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_dir=args.output_dir,
        device=device,
        evaluate_during_training=True,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.dev_batch_size,
        warmup_steps=args.warmup_steps,
        warmup_proportion=args.warmup_proportion,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        save_best_only=args.save_best_only,
        save_best_mode=args.save_best_mode,
        save_best_metric=args.save_best_metric,
        save_best_patience=args.save_best_patience,
        best_metric_minimal_gain=args.best_metric_minimal_gain,
    )
    trainer.train()


if __name__ == "__main__":
    main()
