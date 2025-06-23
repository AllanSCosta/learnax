

import os
import wandb
from omegaconf import OmegaConf

import wandb
import glob
import pickle
import shutil


class Run:
    """Wrapper for wandb run object"""

    def __init__(
        self,
        id: str,
        path: str,
        name: str = None,
        wandb_run=None,
    ):
        self.id = id
        self.path = path
        self.name = name
        self.wandb_run = wandb_run

    def __str__(self):
        return f"Run({self.name})"

    def log(self, data):
        self.wandb_run.log(data)

    @classmethod
    def new(
        cls,
        project: str,
        path: str,
        config: OmegaConf,
        entity: str = "molecular-machines",
    ):
        wandb_path = os.path.join(path, "wandb")
        run = wandb.init(
            project=project,
            dir=os.path.expanduser(wandb_path),
            config=OmegaConf.to_container(config),
            tags=[],
            entity=entity,
        )

        run_path = os.path.join(path, run.id)
        os.makedirs(run_path, exist_ok=False)

        with open(f"{run_path}/config.yml", "w") as f:
            OmegaConf.save(OmegaConf.to_container(config), f)

        return cls(run.id, run_path, run.name, run)

    @classmethod
    def restore(
        cls,
        id: str,
        path: str,
        project: str,
        entity: str = "molecular-machines",
    ):
        wandb_path = path + "/wandb"
        wandb_run = wandb.init(
            project=project,
            id=id,
            dir=os.path.expanduser(wandb_path),
            entity=entity,
            resume="must",
        )
        run_path = os.path.join(path, id)
        return cls(id, run_path, wandb_run.name, wandb_run)

    def get_train_state(self, checkpoint: int = -1):
        if checkpoint == -1:
            checkpoint = "latest"
        with open(f"{self.path}/checkpoints/state_{checkpoint}.pyd", "rb") as f:
            train_state = renamed_load(f)
        return train_state

    def get_weights(self, checkpoint: int = -1):
        train_state = self.get_train_state(checkpoint)
        return train_state.params

    def get_config(self):
        with open(f"{self.path}/config.yml", "r") as f:
            return OmegaConf.load(f)

    def read(
        self,
        filepath: str,
    ):
        path = os.path.join(self.path, filepath)
        with open(path, "rb") as f:
            content = pickle.load(f)
        return content

    def read_all(self, pattern: str):
        contents = []

        search_pattern = self.path + "/**/" + pattern
        files = glob.glob(search_pattern, recursive=True)

        for file in files:
            with open(file, "rb") as f:
                contents.append((file, pickle.load(f)))

        return dict(contents)

    def save(
        self,
        filepath: str,
        content,
    ):
        path = os.path.join(self.path, filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(content, f)

    def clear_dir(self, dirpath: str):
        assert dirpath != ""
        path = os.path.join(self.path, dirpath)
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
