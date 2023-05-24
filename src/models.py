from loguru import logger

from transformers import (
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    XLMRobertaForSequenceClassification,
)

# config, tokenizer, model class
MODELS_MAP = {
    "xlmr": (
        XLMRobertaConfig,
        XLMRobertaTokenizerFast,
        XLMRobertaForSequenceClassification,
    )
}


def get_model(
    model_type: str,
    model_init: bool,
    model_name_or_path: str,
    tokenizer_name: str,
    num_outputs: int,
):

    config_cls, tokenizer_cls, model_cls = MODELS_MAP[model_type]

    if model_init:
        config = config_cls()
        tokenizer = tokenizer_cls()
        logger.info(f"Using tokenizer: {tokenizer}")

    else:
        config = config_cls.from_pretrained(model_name_or_path)
        tokenizer = tokenizer_cls.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path
        )
        logger.info(f"Using tokenizer: {tokenizer}")

    if config.num_labels != num_outputs:
        logger.info(f"Setting model output number to {num_outputs}")
        config.num_labels = num_outputs

    logger.info(f"Using config: {config}")

    if model_init:
        logger.info(f"Loading model class: '{model_cls}', from init")
        model = model_cls(config)
    else:
        logger.info(
            f"Loading model class: '{model_cls}', from path/name: '{model_name_or_path}'"
        )
        model = model_cls.from_pretrained(model_name_or_path, config=config)

    return config, tokenizer, model
