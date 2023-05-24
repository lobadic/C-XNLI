# C-XNLI: Croatian Extension of XNLI Dataset

## Abstract
Comprehensive multilingual evaluations have been encouraged by emerging cross-lingual benchmarks and constrained by existing parallel datasets. To partially mitigate this limitation, we extended the Cross-lingual Natural Language Inference (XNLI) corpus with Croatian. The development and test sets were translated by a professional translator, and we show that Croatian is consistent with other XNLI dubs. The train set is translated using Facebook's 1.2B parameter m2m_100 model. We thoroughly analyze the Croatian train set and compare its quality with the existing machine-translated German set. The comparison is based on 2000 manually scored sentences per language using a variant of the Direct Assessment (DA) score commonly used at the Conference on Machine Translation (WMT). Our findings reveal that a less-resourced language like Croatian is still lacking in translation quality of longer sentences compared to German. However, both sets have a substantial amount of poor quality translations, which should be considered in translation-based training or evaluation setups.

For additional details check out the [paper](https://aclanthology.org/TODO).
## Introduction
- TODO

## Installation & Requirements

- create env:
`conda create --name CUSTOM_NAME python=3.10`
- activate env:
`conda activate CUSTOM_NAME`
- install PyTorch 1.12:
`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
- install other requirements:
`pip install -r requirements.txt`
- install package in editable mode:
`pip install -e .`

## Datasets
- TODO

## Usage

- Train XLMR model on XNLI:
```
TRAIN_FILEPATH=/path/to/dataset/train.csv
DEV_FILEPATH=/path/to/dataset/dev.csv

OUTPUT_DIR=/path/to/output/dir

python3 train.py \
    --gpu "0" \
    \
    --train_filepath $TRAIN_FILEPATH \
    --dev_filepath $DEV_FILEPATH \
    --sentence1_colname "premise" \
    --sentence2_colname "hypo" \
    --label_colname "label" \
    --possible_labels "contradiction" "neutral" "entailment" \
    --task "classification" \
    --max_seq_length 128 \
    \
    --model_type "xlmr" \
    --model_name_or_path "xlm-roberta-base" \
    --num_outputs 3 \                   # number of model outputs
    \
    --optimizer_name "adamw" \
    --optimizer_kwargs '{"learning_rate": 3e-5, "weight_decay": 0.01}' \
    --lr_scheduler_name "linear" \
    \
    --output_dir $OUTPUT_DIR \
    --seed 12345 \
    --num_train_epochs 3 \
    --train_batch_size 32 \
    --dev_batch_size 64 \
    --warmup_proportion 0.06 \          # warmup proportion, here 6% of total training steps will be warmup steps
    --gradient_accumulation_steps 1 \
    --num_workers 8                     # number of dataloader workers

    # Optional:
    # --save_best_only \                # whether to save only the best model (better than all previous) based on `--save_best_metric` metric
    # --save_best_mode "max" \          # save best mode, "max" (default) or "min"
    # --save_best_metric "cls_report" "weighted avg" "f1-score" \ # metric which would serve as a basis for saving the best model;
                                                                  # supports nested objects (argument then requires list of keys to access the desired metric);
                                                                  # argument which accesses `metrics` dictionary returned from `Evaluator` after evaluation
    # --save_best_patience 10 \         # save best patience
```

- Evaluate XLMR model on XNLI:

```
EVAL_FILEPATH=/path/to/dataset/test.csv
MODEL_PATH=/path/to/model/dir/checkpoint-122720

python3 evaluate.py \
    --gpu "0" \
    \
    --eval_filepath $EVAL_FILEPATH \
    --sentence1_colname "premise" \
    --sentence2_colname "hypo" \
    --label_colname "label" \
    --possible_labels "contradiction" "neutral" "entailment" \
    --task "classification" \
    --max_seq_length 128 \
    \
    --model_type "xlmr" \
    --model_path $MODEL_PATH \
    --num_outputs 3 \
    \
    --batch_size 64 \
    --num_workers 8 \
    \
    --write_metrics \                   # writes results to `test_results.txt` inside $MODEL_PATH
    --save_results \                    # writes results to JSON `--results_filename` inside $MODEL_PATH; for easier parsing of results
    --results_filename "xnli_test_results.json"


    # Optional:
    # --save_predictions                # saves predictions inside $MODEL_PATH

```
