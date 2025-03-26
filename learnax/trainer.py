from collections import defaultdict
import os
import pickle
import shutil
import time
from flax import linen as nn
from typing import Callable, NamedTuple, Tuple, Dict, Any

import jax
from jax.tree_util import tree_map, tree_reduce
import jax.numpy as jnp
import numpy as np
import optax
import functools


from wandb.sdk.wandb_run import Run
from torch.utils.data import DataLoader
import torch


from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh, NamedSharding

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



import einops as ein
from .utils import tree_stack, tree_unstack

class Trainer:

    def __init__(
        self,

        model: nn.Module,
        learning_rate,
        losses,
        seed,
        train_dataset,
        num_epochs,
        batch_size,
        num_workers,
        save_every,

        val_every: int = None,
        val_dataset: Any = None,

        save_model: Callable = None,
        run: Run = None,
        registry = None,
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
        self.model = model
        # torch.multiprocessing.set_start_method('spawn')
        self.transform = model

        # self.optimizer = optax.adam(learning_rate, 0.9, 0.999)
        self.optimizer = optax.inject_hyperparams(optax.adam)(learning_rate, 0.9, 0.999)

        self.losses = losses
        self.train_dataset = train_dataset
        self.num_epochs = num_epochs
        self.seed = seed
        self.save_every = save_every

        if registry == None:
            self.registry_path = None
        else:
            self.registry_path = os.environ.get('TRAINAX_REGISTRY_PATH') + '/' + registry

        self.batch_size = batch_size
        self.num_workers = num_workers
        print(f"Batch Size: {self.batch_size}")
        self.save_model = save_model
        self.run = run

        self.name = self.run.name if run else "trainer"
        self.max_grad = 1000.0

        def _make_loader(dataset):
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=lambda x: x,
                shuffle=True,
            )

        self.loaders = {
            'train': _make_loader(train_dataset),
            'val': _make_loader(val_dataset) if val_dataset else None
        }

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

        distribute = True
        self.distribute = distribute
        self.device_count = jax.local_device_count() if self.distribute else 1

        if self.distribute:
            mesh = Mesh(mesh_utils.create_device_mesh((self.device_count,)), axis_names=("ax"))
            self.sharding = NamedSharding(mesh, P("ax"))

    def init(self, train_state=None):
        print("Initializing Model...")
        self.rng_seq = jax.random.key(self.seed)
        self.rng_seq, init_rng = jax.random.split(self.rng_seq)

        def _init(rng, *datum):
            param_rng, _ = jax.random.split(rng)
            params = self.transform.init(param_rng, *datum)["params"]
            opt_state = self.optimizer.init(params)
            return TrainState(
                params,
                opt_state,
            )
        
        if train_state == None:
            init_datum = next(iter(self.loaders['train']))[0]
            init_datum = [init_datum] if type(init_datum) != list else init_datum

            clock = time.time()
            self.train_state = _init(init_rng, *init_datum)
            print("Init Time:", time.time() - clock)
            num_params = sum(
                x.size for x in jax.tree_util.tree_leaves(self.train_state.params)
            )

            print(f"Model has {num_params:.3e} parameters")
            if self.run:
                self.run.log({"model/num_params": num_params})
        else:
            self.train_state = train_state

    def loss(self, params, keys, batch, step):
        def _apply_losses(rng_key, datum: Any):
            model_output = self.transform.apply(
                {"params": params}, *datum, rngs={"params": rng_key}
            )
            return self.losses(rng_key, model_output, datum, step)


        output, loss, metrics = jax.vmap(_apply_losses, in_axes=(0, 0))(keys, batch)

        loss = jnp.where(jnp.isnan(loss), 0.0, loss)
        metrics = {k: v.mean() for k, v in metrics.items()}
        loss = loss.mean()

        return loss, (output, loss, metrics)

    @functools.partial(
        jax.pmap,
        static_broadcasted_argnums=(0,),
        in_axes=(None, None, 0, 0, None),
        axis_name='devices'
    )
    def grad(self, params, keys, batch, step):
        return jax.grad(
            lambda params, rng, batch, step: self.loss(params, rng, batch, step),
            has_aux=True,
        )(params, keys, batch, step)


    def grad_non_pmapped(self, params, keys, batch, step):
        data = tree_unstack(batch)
        keys = tree_unstack(keys)
        grads = []
        metrics = []
        for d, k in zip(data, keys):
            grad, metrics_ = jax.grad(lambda params, rng, batch, step: self.loss(params, rng, batch, step), has_aux=True)(params, k, d, step)
            grads.append(grad)
            metrics.append(metrics_)
        return tree_stack(grads), tree_stack(metrics)

    def update(self, rng, state, batch, step):
        grad, (output, loss, metrics) = self.grad(state.params, rng, batch, step)

        # reduce metrics
        metrics = tree_map(lambda v: jnp.mean(v), metrics)
        loss = jnp.mean(loss)
        metrics = dict(loss=loss, **metrics)

        # reduce gradients
        grad = tree_map(lambda v: jnp.mean(v, axis=0), grad)

        # clip gradients
        grad = tree_map(lambda v: jnp.clip(v, -self.max_grad, self.max_grad), grad)

        # get mean gradient norm for metrics
        mean_abs_grad = tree_map(lambda v: jnp.mean(jnp.linalg.norm(v)), grad)
        mean_grad_norm = tree_reduce(lambda x, y: x + y, mean_abs_grad) / tree_reduce(lambda x, y: x + y, tree_map(lambda v: 1, mean_abs_grad))
        metrics.update({'mean_grad_norm': mean_grad_norm})

        # update parameters
        updates, opt_state = self.optimizer.update(grad, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        return output, TrainState(params, opt_state), metrics

    def epoch(self, epoch):
        for split in self.loaders.keys():
            if self.loaders[split] == None: continue

            if (self.train_only and split != "train") or (
                self.val_every and (split != "train") and (epoch % self.val_every != 0)
            ):
                continue

            loader = self.loaders[split]

            pbar = tqdm(loader, position=1, disable=False)
            pbar.set_description(f"[{self.name}] {split}@{epoch}")

            # batch_size = None
            epoch_metrics = defaultdict(list)

            for step, data in enumerate(pbar):
                if len(data) != self.batch_size:
                    continue

                batch = tree_stack([[d] if type(d)!= list else d for d in data])

                # prepare for pmap
                device_count = self.device_count
                batch_size = self.batch_size

                batch = tree_map(lambda v: ein.rearrange(v, '(p q) ... -> p q ...', p=device_count, q=batch_size // device_count) if not (v is None) else v, batch)
                # batch = jax.device_put(batch, self.sharding)
                # batch = tree_map(lambda v: ein.rearrange(v, 'p q ... -> (p q) ...') if not (v is None) else v, batch)

                total_step = epoch * len(loader) + step

                self.rng_seq, subkey = jax.random.split(self.rng_seq)
                keys = jax.random.split(subkey, len(data))
                keys = ein.rearrange(keys, '(p q) ... -> p q ...', p=device_count, q=batch_size // device_count)

                output, new_train_state, step_metrics = self.update(
                    keys, self.train_state, batch, total_step
                )
                pbar.set_postfix({"loss": f"{step_metrics['loss']:.3e}"})

                _param_has_nan = lambda agg, p: jnp.isnan(p).any() | agg
                has_nan = tree_reduce(
                    _param_has_nan, new_train_state.params, initializer=False
                )

                step_metrics.update(dict(has_nan=has_nan))
                step_metrics.update({'learning_rate': self.train_state.opt_state.hyperparams['learning_rate']})

                if not has_nan and split == "train":
                    self.train_state = new_train_state

                if split == "train":
                    for k, v in step_metrics.items():
                        self.metrics[f"{split}/{k}"].append(float(v))

                    if self.run:
                        self.run.log(
                            {
                                **{
                                    f"{split}/{k}": float(v)
                                    for (k, v) in step_metrics.items()
                                },
                                "step": total_step,
                            }
                        )

                    if self.run and self.registry_path and total_step % self.save_every == 0:
                        registry_path = self.registry_path
                        params_dir_path = os.path.join(registry_path, self.run.id, 'checkpoints')
                        os.makedirs(params_dir_path, exist_ok=True)
                        params_path = os.path.join(params_dir_path, f"state_{total_step}.pyd")
                        with open(params_path, "wb") as file:
                            checkpoint = jax.device_get(self.train_state)
                            pickle.dump(checkpoint, file)

                    if self.run and self.registry_path and total_step % 300 == 0:
                        registry_path = self.registry_path
                        params_dir_path = os.path.join(registry_path, self.run.id, 'checkpoints')
                        os.makedirs(params_dir_path, exist_ok=True)
                        params_path = os.path.join(params_dir_path, f"state_latest.pyd")
                        with open(params_path, "wb") as file:
                            checkpoint = jax.device_get(self.train_state)
                            pickle.dump(checkpoint, file)

                for k, v in step_metrics.items():
                    epoch_metrics[k].append(v)

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

    def train(self) -> Run:
        print("Training...")
        for epoch in tqdm(range(self.num_epochs), position=0):
            self.epoch(epoch=epoch)

    # def run_sample(self, rng_seq, state, batch, step):
    #     keys = jax.random.split(next(rng_seq), min(9, len(batch)))

    #     if hasattr(self, 'sample_params'):
    #         params_ = {**state.params, **self.sample_params}
    #     else:
    #         params_ = state.params

    #     batched = inner_stack(batch[:9])

    #     start = time.time()
    #     samples, trajectories = jax.vmap(
    #         self.sample_model,
    #         in_axes=(None, 0, 0)
    #     )(params_, keys, batched)
    #     end = time.time()
    #     samples = inner_split(samples)

    #     sample_metrics = {}
    #     if step != 0:
    #         sample_metrics.update({'sample_time': end - start})

    #     if self.plot_mode == 'samples':
    #         self.sample_plot(self.run, samples, None)

    #     elif self.plot_mode == 'trajectory':
    #         trajectories = [inner_split(traj) for traj in inner_split(trajectories)]
    #         self.sample_plot(self.run, trajectories, None)

    #     if self.sample_metrics:
    #         sample_metrics_ = defaultdict(list)
    #         for sample, batch in zip(samples, batch):
    #             sample_metrics = self.sample_metrics(sample, batch)
    #             for k, v in sample_metrics.items():
    #                 sample_metrics_[k].append(v)
    #         sample_metrics.update({k: np.mean(v) for k, v in sample_metrics_.items()})

    #     return sample_metrics
