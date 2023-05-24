from loguru import logger
import time
from typing import Callable, Optional, Any, Dict

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)


def default_prepare_inputs_fn(batch, device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}


def _default_evaluation_loop(
    model,
    prepare_inputs_fn: Callable,
    dataloader: torch.utils.data.DataLoader,
    num_labels: int,
    device: torch.device,
):

    num_samples = len(dataloader.dataset)
    preds_shape = (num_samples, num_labels)

    preds = np.zeros(preds_shape)
    labels = np.zeros(num_samples)

    t0 = time.perf_counter()

    i = 0

    model.eval()

    for batch in tqdm(dataloader, desc="Evaluating"):

        inputs = prepare_inputs_fn(batch, device=device)
        label = inputs["labels"]

        with torch.no_grad():

            loss, outputs = model(**inputs, return_dict=False)
            j = outputs.shape[0]

            preds[i : i + j] = outputs.cpu().numpy()
            labels[i : i + j] = label.cpu().numpy()

            i += j

    t1 = time.perf_counter()
    delta_t = t1 - t0

    logger.info(
        f"Predicted {len(preds)} examples in {delta_t} seconds, {len(preds)/(delta_t)} per second"
    )

    return preds, labels, delta_t


class ClassificationEvaluator:
    def __init__(self, **kwargs):
        self.preds, self.labels = None, None

    def _prepare_preds(self, preds: np.ndarray) -> np.ndarray:
        return np.argmax(preds, axis=1)

    def evaluation_loop(
        self,
        model,
        dataloader: torch.utils.data.DataLoader,
        num_labels: int,
        device: torch.device,
        **kwargs,
    ):
        if hasattr(model, "prepare_inputs") and callable(model.prepare_inputs):
            prepare_inputs_fn = model.prepare_inputs
        else:
            prepare_inputs_fn = default_prepare_inputs_fn

        self.preds, self.labels, _ = _default_evaluation_loop(
            model=model,
            prepare_inputs_fn=prepare_inputs_fn,
            dataloader=dataloader,
            device=device,
            num_labels=num_labels,
        )

    def get_metrics(
        self,
        preds: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        if (preds is None) or (labels is None):
            preds, labels = self.preds, self.labels

        preds = self._prepare_preds(preds)

        cm = confusion_matrix(labels, preds).tolist()
        cls_report_d = classification_report(labels, preds, digits=4, output_dict=True)
        cls_report_str = classification_report(
            labels, preds, digits=4, output_dict=False
        )

        metrics = {}
        metrics["confusion_matrix"] = cm
        metrics["cls_report_print"] = cls_report_str
        metrics["cls_report"] = cls_report_d

        return metrics

    def _remap_labels(self, preds: np.ndarray, idx_to_label_map: dict):
        preds_remaped = []
        for pred in preds:
            preds_remaped.append(idx_to_label_map[pred])

        return np.array(preds_remaped)

    def save_predictions(
        self,
        save_filepath: str,
        data_filepath: str,
        predictions_colname: str = "preds",
        idx_to_label_map: dict = {},
        preds: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ):
        assert len(idx_to_label_map)

        if preds is None or labels is None:
            assert (self.preds is not None) and (self.labels is not None)
            preds = self._remap_labels(
                self._prepare_preds(self.preds), idx_to_label_map=idx_to_label_map
            )
            labels = self.labels

        data_df = pd.read_csv(data_filepath)
        data_df[predictions_colname] = preds

        logger.info(f"Saving predictions to {save_filepath}")
        data_df.to_csv(save_filepath, index=False)


def get_evaluator(task: str, **kwargs):

    if task == "classification":
        return ClassificationEvaluator(**kwargs)
    else:
        raise NotImplementedError(
            f'Evaluator for the task "{task}" is not implemented.'
        )
