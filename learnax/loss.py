from types import NoneType
from typing import Dict, List, Tuple, Callable, Optional

from jax import numpy as jnp
import re

from jaxtyping import Array, PyTree

class LossFunction:

    def __init__(self,
        weight: float = 1.0,
        scheduler: Optional[Callable] = None
    ):
        self.weight = weight
        self.scheduler = scheduler

    def _call(
        self, model_output: PyTree, ground_truth: PyTree
    ) -> Tuple[PyTree, Array, Dict[str, float]]:
        raise NotImplementedError

    def __call__(
        self,
        rng_key,
        model_output: PyTree,
        batch: PyTree,
        step: int,
    ) -> Tuple[PyTree, Array, Dict[str, float]]:
        output, loss, metrics = self._call(rng_key, model_output, batch)
        if self.scheduler is not None:
            scheduler_weight = self.scheduler(step)
            loss = loss * scheduler_weight
            loss_name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()
            metrics[loss_name + "_scheduler"] = scheduler_weight
        return output, self.weight * loss, metrics


class LossPipe:

    def __init__(self, loss_list: List[LossFunction]):
        self.loss_list = loss_list

    def __call__(
        self,
        rng_key,
        model_output: PyTree,
        batch: PyTree,
        step: int = 0,
    ):
        loss = 0.0
        metrics = {}
        for loss_fn in self.loss_list:
            model_output, loss_fn_loss, loss_fn_metrics = loss_fn(
                rng_key, model_output, batch, step
            )
            loss += loss_fn_loss
            metrics.update(loss_fn_metrics)
        return model_output, loss, metrics
