Learnax - JAX-based Training Framework
=====================================

Learnax is a powerful training framework built on JAX for accelerated machine learning research. It provides a clean, modular approach to training neural networks with built-in support for distributed computing, checkpointing, and experiment tracking.

.. toctree::
   :maxdepth: 4

   self

Key Features
-----------

- **JAX-based acceleration**: Leverage the power of JAX's transformations and GPU/TPU acceleration
- **Distributed training**: Automatic parallel training across multiple devices using ``jax.pmap``
- **Robust checkpointing**: Flexible checkpoint saving and resuming with automatic management
- **Experiment tracking**: Seamless integration with Weights & Biases for experiment logging
- **Modular architecture**: Clean abstractions for models, losses, training, and evaluation

Core Components
-------------

Trainer
~~~~~~~

The ``Trainer`` class is the central component of Learnax. It orchestrates the training process, handling:

- Model initialization
- Training loop execution with gradient updates
- Metrics tracking
- Checkpointing
- Distributed training across multiple devices

Example usage::

    trainer = Trainer(
        model=my_model,
        learning_rate=1e-4,
        losses=my_loss_pipe,
        seed=42,
        train_dataset=train_data,
        num_epochs=100,
        batch_size=32,
        num_workers=4,
        run=wandb_run
    )

    trainer.init()
    trainer.train()

Checkpointer
~~~~~~~~~~~

The ``Checkpointer`` class manages model checkpoints, providing:

- Automatic saving of periodic checkpoints
- Retaining the best checkpoints based on metrics
- Limiting the number of checkpoints to save disk space
- Simple APIs for loading and resuming training

Example usage::

    checkpointer = Checkpointer(
        registry_path="./experiments",
        run_id="experiment_1",
        save_every=100,
        checkpoint_every=1000,
        keep_best=True,
        metric_name="val/loss"
    )

    # Save checkpoint
    checkpointer.maybe_save_checkpoints(train_state, step=1000, metrics={"val/loss": 0.1})

    # Load checkpoint
    checkpoint_data = checkpointer.load_latest()

Loss System
~~~~~~~~~~

Learnax provides a modular loss system with:

- ``LossFunction``: Base class for individual loss components
- ``LossPipe``: Combines multiple loss functions with weights and scheduling

Example::

    loss_pipe = LossPipe([
        MSELoss(weight=1.0),
        RegularizationLoss(weight=0.01, scheduler=lambda step: min(1.0, step/1000))
    ])

Experiment Registry
~~~~~~~~~~~~~~~~~

The ``Registry`` and ``Run`` classes help manage experiments:

- Track configurations, metrics, and results
- Restore previous runs for analysis or continued training
- Integrate with Weights & Biases for enhanced visualization

Installation
-----------

Install Learnax via pip::

    pip install learnax

Requirements:

- JAX >= 0.3.0
- Flax >= 0.5.0
- Optax >= 0.1.0
- Weights & Biases (optional)

Getting Started
-------------

For a quick start, see the following examples:

- Basic model training
- Distributed training
- Custom loss functions
- Checkpoint management
- Experiment tracking

API Reference
-----------

For detailed API documentation, see the following sections:

- :mod:`learnax.trainer`
- :mod:`learnax.checkpointer`
- :mod:`learnax.loss`
- :mod:`learnax.registry`

* :ref:`genindex`
