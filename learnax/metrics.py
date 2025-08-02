from types import NoneType
from typing import Dict, List, Tuple, Callable, Optional

from jax import numpy as jnp
import re

from jaxtyping import Array, PyTree


class Metric:

    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform

    def _call(
        self,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def __call__(
        self,
        *args,
        **kwargs
    ) -> Dict[str, float]:

        if self.transform is not None:
            model_output = self.transform(*args)
        return self._call(*args, **kwargs)


class MetricPipe:

    def __init__(
        self,
        metrics_list: List[Metric]
    ):
        self.metrics_list = metrics_list

    def __call__(
        self,
        *args,
        **kwargs
    ):
        metrics = {}
        for metric in self.metrics_list:
            metric_dict = metric(*args, **kwargs)
            metrics.update(metric_dict)
        return metrics
