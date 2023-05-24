# CXNLI-paper

Croatian extension of The Cross-Lingual NLI Corpus (XNLI).
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