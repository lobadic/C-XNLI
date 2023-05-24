import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

from src.data import ClassificationDataset
from src.evaluators import get_evaluator
from src.models import get_model
from src.utils import save_results, write_eval_results


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", required=True, type=str, help=".")

    # DATA ARGS:
    parser.add_argument(
        "--eval_filepath", required=True, type=str, help="Path to eval dataset."
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
        "--possible_labels",
        required=False,
        type=str,
        nargs="+",
        help="Possible labels. if classification task",
    )
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        choices=["regression", "classification"],
        help="Training task.",
    )

    # MODEL ARGS:
    parser.add_argument(
        "--model_type",
        type=str,
        default="",
        help="Model type.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="",
        help="Name of the tokenizer.",
    )
    parser.add_argument(
        "--num_outputs",
        type=int,
        required=True,
        help="Number of model outputs.",
    )
    parser.add_argument(
        "--max_seq_length",
        required=False,
        default=128,
        type=int,
        help="Name of the tokenizer.",
    )

    # OTHER ARGS:
    parser.add_argument(
        "--batch_size",
        type=int,
        default="",
        help="Batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default="",
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--write_metrics",
        action="store_true",
        help="Whether to write metrics to `test_results.txt` inside model_path dir.",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Whether to save results (metrics/kpis of interest) to json later for easier parsing and reporting.",
    )
    parser.add_argument(
        "--results_filename",
        type=str,
        default="",
        help="Filename under which results of interest will be saved. inside `model_path` dir. json file",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save model's predictions to csv.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    config, tokenizer, model = get_model(
        model_type=args.model_type,
        model_init=None,
        model_name_or_path=args.model_path,
        tokenizer_name=args.tokenizer_name,
        num_outputs=args.num_outputs,
    )
    model.to(device)
    model.eval()

    eval_dataset = ClassificationDataset(
        filepath=args.eval_filepath,
        text_columns=[args.sentence1_colname, args.sentence2_colname],
        label_column=args.label_colname,
        possible_labels=args.possible_labels,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    sampler = SequentialSampler(eval_dataset)
    dl = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    if args.task == "classification":
        assert args.num_outputs == len(args.possible_labels)

    evaluator = get_evaluator(task=args.task)
    evaluator.evaluation_loop(
        model=model,
        dataloader=dl,
        num_labels=args.num_outputs,
        device=device,
    )

    metrics = evaluator.get_metrics()

    for k, v in metrics.items():
        print(k)
        print(v)

    if args.write_metrics:
        write_eval_results(
            outpath=os.path.join(args.model_path, "test_results.txt"),
            eval_filepath=args.eval_filepath,
            metrics=metrics,
            other_params={"Max seq length": args.max_seq_length},
        )

    if args.save_results:
        assert args.results_filename

        results_filepath = os.path.join(args.model_path, args.results_filename)

        save_results(
            results_filepath,
            eval_filepath=args.eval_filepath,
            metrics=metrics,
            task=args.task,
        )

    if args.save_predictions:
        evaluator.save_predictions(
            save_filepath=os.path.join(
                args.model_path, Path(args.eval_filepath).name[:-4] + "_predictions.csv"
            ),
            data_filepath=args.eval_filepath,
            predictions_colname="preds",
            idx_to_label_map=eval_dataset.idx_to_label_map
            if args.task == "classification"
            else {},
        )


if __name__ == "__main__":
    main()
