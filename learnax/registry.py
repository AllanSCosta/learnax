import yaml
import logging
import os
import shutil
from pathlib import Path

# import git

import wandb


# logger = logging.getLogger("aim.sdk.reporter")
# logger.setLevel(logging.WARNING)


from omegaconf import OmegaConf
import pickle

import glob


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed = module
        if "trainax" in module:
            renamed = module.replace("trainax", "learnax")
        return super(RenameUnpickler, self).find_class(renamed, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


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
