from typing import Optional
from torch.optim import Adam, AdamW, SGD


def get_optimizer(
    optimizer_name: str,
    model,
    learning_rate: float,
    weight_decay: Optional[float] = None,
    **optimizer_kwargs,
):

    no_decay = ["bias", "LayerNorm.weight"]

    if weight_decay is not None:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = [{"params": model.parameters()}]

    if optimizer_name == "adam":
        optimizer = Adam(
            optimizer_grouped_parameters, lr=learning_rate, **optimizer_kwargs
        )
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **optimizer_kwargs,
        )
    elif optimizer_name == "sgd":
        optimizer = SGD(
            optimizer_grouped_parameters, lr=learning_rate, **optimizer_kwargs
        )
    else:
        raise NotImplementedError(f"Optimizer '{optimizer_name}' is not implemented.")

    return optimizer
