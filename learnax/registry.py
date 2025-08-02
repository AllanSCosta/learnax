import yaml
import logging
import os
import shutil
from pathlib import Path

# import git

import wandb

from omegaconf import OmegaConf

from .run import Run

class Registry:
    """
    Abstract collection of models
    """

    def __init__(
        self,
        project: str,
        base_path: str = None,
    ):

        if base_path is None:
            base_path = os.getenv("TRAINAX_REGISTRY_PATH")
        os.makedirs(base_path, exist_ok=True)

        path = os.path.join(base_path, project)
        os.makedirs(path, exist_ok=True)
        self.project = project
        self.path = Path(path)

    def new_run(
        self,
        config: OmegaConf,
    ) -> Run:
        return Run.new(
            project=self.project,
            path=self.path,
            config=config,
        )

    def restore_run(
        self,
        id: str,
        read_only=False,
    ) -> Run:
        return Run.restore(
            id=id,
            path=str(self.path),
            project=self.project,
        )

    def fetch_run(
        self,
        id: str,
    ) -> Run:
        run_path = os.path.join(self.path, id)
        return Run(
            id=id,
            path=run_path,
        )
