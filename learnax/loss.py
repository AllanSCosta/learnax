from types import NoneType
from typing import Dict, List, Tuple, Callable, Optional

from jax import numpy as jnp
import re

from jaxtyping import Array, PyTree

class LossFunction:

    def __init__(
        self,
        weight: float = 1.0,
        scheduler: Optional[Callable] = lambda x: 1.0
    ):
        self.weight = weight
        self.scheduler = scheduler

    def _call(
        self, model_output: PyTree, ground_truth: PyTree
    ) -> Tuple[PyTree, Array, Dict[str, float]]:
        raise NotImplementedError

    def __call__(
        self,
        model_output: PyTree,
        batch: PyTree,
        step: int = 0,
    ) -> Tuple[PyTree, Array, Dict[str, float]]:
        output, loss, metrics = self._call(model_output, batch)
        if self.scheduler is not None:
            scheduler_weight = self.scheduler(step)
            loss = loss * scheduler_weight
            # loss_name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()
            # metrics[loss_name + "_scheduler"] = scheduler_weight
        return output, self.weight * loss, metrics


class LossPipe:

    def __init__(self,
        loss_list: List[LossFunction],
        transform: Optional[Callable] = None):
        self.loss_list = loss_list
        self.transform = transform

    def __call__(
        self,
        model_output: PyTree,
        batch: PyTree,
        step: int = 0,
    ):
        loss = 0.0
        metrics = {}

        if self.transform is not None:
            model_output = self.transform(model_output)

        for loss_fn in self.loss_list:
            model_output, loss_fn_loss, loss_fn_metrics = loss_fn(model_output, batch)
            loss += loss_fn_loss
            metrics.update(loss_fn_metrics)

        return model_output, loss, metrics
