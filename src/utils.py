import os
from pathlib import Path

import random
import numpy as np
import torch

from io_helpers import load_json, save_json


def write_eval_results(
    outpath: str, eval_filepath: str, metrics: dict, other_params: dict
):

    if Path(outpath).exists():
        out_mode = "a"
    else:
        out_mode = "w"

    # writing logs
    with open(outpath, out_mode) as writer:
        writer.write(f"\nEvaluating on the {os.path.abspath(eval_filepath)} dataset.\n")

        for k, v in other_params.items():
            writer.write(f"{k} = {v}\n")

        for key in metrics.keys():
            writer.write("%s = %s\n" % (key, str(metrics[key])))


def save_results(outpath: str, eval_filepath: str, metrics: dict, task: str):

    if task == "classification":
        results_out = {eval_filepath: metrics["cls_report"]}
    else:
        results_out = {eval_filepath: metrics}

    if Path(outpath).exists() and Path(outpath).is_file():
        data = load_json(outpath)
        data.update(results_out)
        save_json(outpath, data=data)

    else:
        save_json(outpath, data=results_out)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def multilevel_dict_lookup(d: dict, keys: list):
    # d: dict
    #   - nested dict object
    # keys: list
    #   - list of lookup keys for each level

    for k in keys:
        d = d.get(k, None)

        assert d is not None
    return d
