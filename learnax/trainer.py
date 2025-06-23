from collections import defaultdict
import os
import time
from flax import linen as nn
from typing import List, Callable, NamedTuple, Any, Dict, Optional

import jax
from jax.tree_util import tree_map, tree_reduce
import jax.numpy as jnp
import numpy as np
import optax
import functools

from wandb.sdk.wandb_run import Run
from torch.utils.data import DataLoader

import einops as ein
from .utils import tree_stack, tree_unstack
from .checkpointer import Checkpointer

def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class TrainState(NamedTuple):
    params: Any
    opt_state: Any
    step: int = 0


import einops as ein
from .utils import tree_stack, tree_unstack


from ctypes import cdll
libcudart = cdll.LoadLibrary('libcudart.so')

class Trainer:

    """Trainer class for training Flax models with batched data processing and distributed training support.

    This trainer handles:
    - Model initialization and parameter management
    - Training loop execution with gradient updates
    - Checkpointing calls and model saving
    - Metrics tracking and logging through wandb
    - Distributed training with jax.pmap
    - Distribution of batches with jax.vmap
    - Validation dataset evaluation
    - Optional model visualization and sampling

    The trainer supports configurable training schedules, gradient clipping, and
    various debugging options like single batch or single datum training.

    """


    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        losses: List[Callable],
        seed: int,
        train_dataset: List,
        num_epochs : int,
        batch_size : int,
        num_workers : int,

        save_every: int = 300,
        checkpoint_every: int = 1000,
        max_checkpoints: int = 5,
        keep_best_checkpoint: bool = False,
        best_metric_name: str = "val/loss",
        minimize_metric: bool = True,
        load_checkpoint: Optional[int] = None,

        val_every: int = None,
        val_datasets: List = None,
        save_model: Callable = None,
        run: Run = None,
        registry=None,
        compile: bool = False,
        single_datum: bool = False,
        single_batch: bool = False,
        train_only: bool = False,
        plot_pipe: Callable = None,
        plot_every: int = 1000,
        plot_model: Callable = None,
        # plot_metrics: MetricsPipe = None,
        load_weights: bool = False,
        sample_every: int = None,
        sample_model: Callable = None,
        sample_params: str = None,
        sample_plot: Callable = None,
        sample_batch_size=None,
        sample_metrics=None,
    ):

        """
        Initialize the Trainer class.

        Args:
            model (nn.Module): The model to train.
            transform (Callable): The transformation to apply to the data.
            learning_rate (float): The learning rate for the optimizer.
            losses (List[Callable]): The loss functions to use.
            train_dataset (Dataset): The training dataset.
            num_epochs (int): The number of epochs to train for.
            batch_size (int): The batch size for training.
            num_workers (int): The number of workers for data loading.
            save_every (int): The frequency to save the model.
            checkpoint_every (int): The frequency to save a checkpoint.
            val_every (int): The frequency to validate the model.
            val_datasets (Dataset): The validation dataset.
            save_model (Callable): The function to save the model.
            run (Run): The run object.
        """

        self.model = model
        self.transform = model

        self.optimizer = optax.inject_hyperparams(optax.adam)(learning_rate, 0.9, 0.999)

        self.losses = losses
        self.train_dataset = train_dataset
        self.num_epochs = num_epochs
        self.seed = seed
        if registry == None:
            self.registry_path = None
        else:
            self.registry_path = (
                os.environ.get("TRAINAX_REGISTRY_PATH") + "/" + registry
            )

        # Initialize checkpointer if registry path and run are available
        self.checkpointer = None
        if self.registry_path and run:
            self.checkpointer = Checkpointer(
                registry_path=self.registry_path,
                run_id=run.id,
                save_every=save_every,
                checkpoint_every=checkpoint_every,
                max_to_keep=max_checkpoints,
                keep_best=keep_best_checkpoint,
                metric_name=best_metric_name,
                minimize_metric=minimize_metric,
                prepare_fn=jax.device_get
            )

        self.batch_size = batch_size
        self.num_workers = num_workers
        print(f"Batch Size: {self.batch_size}")
        self.save_model = save_model
        self.run = run

        self.name = self.run.name if run else "trainer"
        self.max_grad = 0.1

        def _make_loader(dataset):
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=lambda x: x,
            )

        self.loaders = {
            "train": _make_loader(train_dataset),
        }

        if val_datasets != None:
            for val_name, val in val_datasets.items():
                self.loaders[val_name] = _make_loader(val)

        self.train_only = train_only
        self.single_batch = single_batch
        self.single_datum = single_datum

        if self.single_batch:
            print("[!!WARNING!!] using single batch")
            sample_batch = next(iter(self.loaders["train"]))
            self.loaders = {"train": [sample_batch] * 1000}

        elif self.single_datum:
            print("[!!WARNING!!] using single datum")
            sample_batch = next(iter(self.loaders["train"]))
            sample_datum = sample_batch[0]
            sample_batch = [sample_datum] * self.batch_size
            self.loaders = {"train": [sample_batch] * 1000}

        self.plot_pipe = plot_pipe
        self.plot_every = plot_every
        self.plot_model = plot_model
        # self.plot_metrics = plot_metrics

        self.val_every = val_every
        self.load_weights = load_weights

        self.sample_every = sample_every
        self.metrics = defaultdict(list)

        self.device_count = jax.local_device_count()

    def init(self, train_state=None):
        """
        Initialize the trainer.

        Args:
            train_state (TrainState, optional): The initial training state. Defaults to None.
        """
        print("Initializing Model...")
        # Create a random number generator key from the seed
        self.rng_seq = jax.random.key(self.seed)
        # Split the key to use one for initialization and keep the other for future use
        self.rng_seq, init_rng = jax.random.split(self.rng_seq)

        def _init(rng, *datum):
            """Helper function to initialize model parameters and optimizer state."""
            # Split RNG key for parameter initialization
            param_rng, _ = jax.random.split(rng)
            # Initialize model parameters with a sample datum
            params = self.transform.init(param_rng, *datum)["params"]
            # Initialize optimizer state with the parameters
            opt_state = self.optimizer.init(params)
            # Return the training state
            return TrainState(
                params,
                opt_state,
            )

        # Try loading from checkpoint if requested
        if train_state is None and load_checkpoint is not None and self.checkpointer:
            checkpoint_data = self.checkpointer.load_checkpoint(load_checkpoint)
            if checkpoint_data:
                print(f"Loaded checkpoint from step {checkpoint_data['step']}")
                train_state = checkpoint_data['state']

        if train_state is None:
            # Get a sample from the training data for initialization
            init_datum = next(iter(self.loaders["train"]))[0]
            # Ensure init_datum is a list for unpacking with *
            init_datum = [init_datum] if type(init_datum) != list else init_datum

            # Measure initialization time
            clock = time.time()
            self.train_state = _init(init_rng, *init_datum)
            print("Init Time:", time.time() - clock)

            # Calculate and log the total number of parameters
            num_params = sum(
                x.size for x in jax.tree_util.tree_leaves(self.train_state.params)
            )

            print(f"Model has {num_params:.3e} parameters")
            if self.run:
                self.run.log({"model/num_params": num_params})
        else:
            # Use the provided training state
            self.train_state = train_state

    def loss(self, params, keys, batch):
        """
        Calculate the loss and metrics for a batch of data.

        This method:
        1. Applies the model to each datum in the batch (using vmap)
        2. Computes losses and metrics for each prediction
        3. Handles NaN values in the loss
        4. Returns the mean loss and metrics

        Args:
            params: Model parameters
            keys: RNG keys for each datum in the batch
            batch: Batch of data to process

        Returns:
            A tuple containing:
            - Mean loss value
            - Tuple of (model outputs, loss, metrics)
        """
        def _apply_losses(rng_key, datum: Any):
            # Apply the model to a single datum using the given parameters
            model_output = self.transform.apply(
                {"params": params}, *datum, rngs={"params": rng_key}
            )
            # Calculate losses and metrics for this datum
            return self.losses(rng_key, model_output, datum, 0)

        # Apply the loss function to each datum in the batch using vmap
        output, loss, metrics = jax.vmap(_apply_losses, in_axes=(0, 0))(keys, batch)

        # Replace NaN values with zeros in the loss to prevent training instability
        loss = jnp.where(jnp.isnan(loss), 0.0, loss)
        # Average metrics across the batch
        metrics = {k: v.mean() for k, v in metrics.items()}
        # Average loss across the batch
        loss = loss.mean()

        return loss, (output, loss, metrics)

    @functools.partial(
        jax.pmap,
        static_broadcasted_argnums=(0,),
        in_axes=(None, None, 0, 0, None),
        axis_name="devices",
    )
    def grad(self, params, keys, batch):
        """
        Compute gradients for the loss function with respect to parameters.

        This is a parallel mapped (pmap) function that computes gradients across
        multiple devices. It uses JAX's automatic differentiation to calculate
        gradients of the loss with respect to model parameters.

        Args:
            params: Model parameters
            keys: RNG keys for stochasticity
            batch: Batch of data distributed across devices

        Returns:
            Tuple of (gradients, auxiliary outputs from loss)
        """
        return jax.grad(
            lambda params, rng, batch: self.loss(params, rng, batch),
            has_aux=True,
        )(params, keys, batch)

    def update(self, rng, state, batch):
        """
        Update model parameters based on computed gradients.

        This method:
        1. Computes gradients using the grad() method
        2. Averages metrics and losses across devices
        3. Clips gradients to prevent exploding gradients
        4. Updates parameters using the optimizer
        5. Returns the updated training state and metrics

        Args:
            rng: Random number generator key
            state: Current TrainState (params, opt_state, step)
            batch: Batch of data to process

        Returns:
            Tuple of (model_output, new_train_state, metrics)
        """
        # Compute gradients across all devices
        grad, (output, loss, metrics) = self.grad(state.params, rng, batch)

        # Average metrics across devices
        metrics = tree_map(lambda v: jnp.mean(v), metrics)
        loss = jnp.mean(loss)
        metrics = dict(loss=loss, **metrics)

        # Average gradients across devices
        grad = tree_map(lambda v: jnp.mean(v, axis=0), grad)

        # Clip gradients to prevent explosions
        grad_leaves, _ = jax.tree.flatten(grad)
        norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
        normalize = lambda g: jnp.where(norm < self.max_grad, g, g * (self.max_grad / norm))
        grad = jax.tree.map(normalize, grad)

        # Add gradient norm to metrics for monitoring
        metrics.update({"gradient norm": norm})

        # Update parameters using the optimizer
        updates, opt_state = self.optimizer.update(grad, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        # Return model output, updated state, and metrics
        return output, TrainState(params, opt_state, state.step + 1), metrics

    def epoch(self, epoch):
        """
        Run a single epoch of training or validation.

        This method processes all data splits (train, validation, etc.) for a single epoch:
        1. Iterates through batches of each data loader
        2. Processes each batch with the model
        3. Updates parameters for training splits
        4. Logs metrics and saves checkpoints
        5. Handles NaN parameters and skips problematic updates

        Args:
            epoch: Current epoch number
        """
        for split in self.loaders.keys():
            # Skip empty loaders
            if self.loaders[split] == None:
                continue

            # Skip validation if train_only is set or if it's not a validation epoch
            if (self.train_only and split != "train") or (
                self.val_every and (split != "train") and (epoch % self.val_every != 0)
            ):
                continue

            loader = self.loaders[split]

            # Set up progress bar for this split
            pbar = tqdm(loader, position=1, disable=False)
            pbar.set_description(f"[{self.name}] {split}@{epoch}")

            epoch_metrics = defaultdict(list)

            for step, data in enumerate(pbar):
                # Skip incomplete batches
                if len(data) != self.batch_size:
                    continue

                # Stack batch data for processing
                batch = tree_stack([[d] if type(d) != list else d for d in data])

                # Prepare batch for parallel processing across devices
                device_count = self.device_count
                batch_size = self.batch_size

                # Reshape batch for parallelism (pmap): distribute batch elements across devices
                batch = tree_map(
                    lambda v: (
                        ein.rearrange(
                            v,
                            "(p q) ... -> p q ...",
                            p=device_count,
                            q=batch_size // device_count,
                        )
                        if not (v is None)
                        else v
                    ),
                    batch,
                )

                # Generate random keys for this batch
                self.rng_seq, subkey = jax.random.split(self.rng_seq)
                keys = jax.random.split(subkey, len(data))
                keys = ein.rearrange(
                    keys,
                    "(p q) ... -> p q ...",
                    p=device_count,
                    q=batch_size // device_count,
                )

                # Time the update step for performance monitoring
                bp_start_time = time.time()
                _, new_train_state, step_metrics = self.update(
                    keys, self.train_state, batch
                )
                bp_end_time = time.time()

                # Show loss in progress bar
                pbar.set_postfix({"loss": f"{step_metrics['loss']:.3e}"})
                # Track update time in metrics
                step_metrics.update({"update_time": bp_end_time - bp_start_time})

                # Check for NaN values in parameters
                _param_has_nan = lambda agg, p: jnp.isnan(p).any() | agg
                has_nan = tree_reduce(
                    _param_has_nan, new_train_state.params, initializer=False
                )

                # Add NaN check and learning rate to metrics
                step_metrics.update(dict(has_nan=has_nan))
                step_metrics.update(
                    {
                        "learning_rate": self.train_state.opt_state.hyperparams[
                            "learning_rate"
                        ]
                    }
                )

                # Update train state if no NaNs and in training split
                if not has_nan and split == "train":
                    self.train_state = new_train_state
                elif has_nan:
                    # Debug which parameters contain NaN values
                    for k, v in new_train_state.params.items():
                        if jnp.isnan(v).any():
                            print(f"Parameter {k} has NaN values, skipping update.")

                # Log metrics for training split
                if split == "train":
                    for k, v in step_metrics.items():
                        self.metrics[f"{split}/{k}"].append(float(v))

                    # Log step metrics to wandb
                    if self.run:
                        self.run.log(
                            {
                                **{
                                    f"{split}/{k}": float(v)
                                    for (k, v) in step_metrics.items()
                                },
                                "step": self.train_state.step,
                            }
                        )

                    # Handle checkpointing with the Checkpointer class
                    if self.checkpointer:
                        # Collect metrics for checkpointing
                        checkpoint_metrics = {
                            f"{split}/{k}": float(v) for k, v in step_metrics.items()
                        }
                        # Let the checkpointer handle all checkpoint logic
                        saved_paths = self.checkpointer.maybe_save_checkpoints(
                            self.train_state,
                            self.train_state.step,
                            checkpoint_metrics
                        )
                        if saved_paths:
                            paths_str = ", ".join(f"{k}" for k in saved_paths.keys())
                            print(f"Saved checkpoints: {paths_str}")

                # Collect metrics for this epoch
                for k, v in step_metrics.items():
                    epoch_metrics[k].append(v)

            # Calculate and log epoch-level metrics
            for k, v in epoch_metrics.items():
                self.metrics[f"{split}_epoch/{k}"].append(float(np.mean(v)))
                if self.run:
                    self.run.log(
                        {
                            **{
                                f"{split}_epoch/{k}": float(np.mean(v))
                                for (k, v) in epoch_metrics.items()
                            },
                            "epoch": epoch,
                        },
                    )

    def train(self, resume_from_checkpoint: bool = False):
        """
        Run the full training loop for the configured number of epochs.

        This method:
        1. Optionally loads the latest checkpoint if resume_from_checkpoint is True
        2. Iterates through the specified number of epochs
        3. Calls the epoch() method for each epoch
        4. Returns the wandb run object if available

        Args:
            resume_from_checkpoint (bool, optional): Whether to resume training from
                the latest checkpoint. Defaults to False.

        Returns:
            The wandb Run object associated with this training run
        """
        if resume_from_checkpoint and self.checkpointer:
            checkpoint_data = self.checkpointer.load_latest()
            if checkpoint_data:
                print(f"Resuming training from step {checkpoint_data['step']}")
                self.train_state = checkpoint_data['state']
            else:
                print("No checkpoint found for resuming training.")

        print("Training...")
        for epoch in tqdm(range(self.num_epochs), position=0):
            self.epoch(epoch=epoch)

        return self.run
