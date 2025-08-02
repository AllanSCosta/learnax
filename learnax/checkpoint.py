import os
import pickle
import jax
from typing import Any, Callable, Dict, Optional, Union
import time
from functools import partial

class Checkpointer:
    """
    A class for managing model checkpoints during training.

    This class handles the saving, loading, and management of checkpoints,
    providing a simple interface that abstracts away the details of file management,
    serialization, and device handling.

    Attributes:
        registry_path (str): Base directory for saving checkpoints
        run_id (str): Identifier for the current run
        checkpoint_dir (str): Full path to the checkpoint directory
        save_every (int): Steps between saving regular checkpoints
        checkpoint_every (int): Steps between saving numbered checkpoints
        max_to_keep (int): Maximum number of numbered checkpoints to retain
        keep_best (bool): Whether to retain the best checkpoint based on a metric
        metric_name (str): Name of the metric to use for best checkpoint selection
        minimize_metric (bool): Whether the metric should be minimized (True) or maximized (False)
        best_metric_value (float): Current best metric value
        prepare_fn (Callable): Function to prepare state for checkpointing
    """

    def __init__(
        self,
        registry_path: str,
        run_id: str,
        save_every: int = 300,
        checkpoint_every: int = 5000,
        max_to_keep: Union[int, float] = None,
        keep_best: bool = False,
        metric_name: str = "val/loss",
        minimize_metric: bool = True,
        prepare_fn: Callable = None
    ):
        """
        Initialize the Checkpointer.

        Args:
            registry_path (str): Base directory for saving checkpoints
            run_id (str): Identifier for the current run
            save_every (int): Number of steps between saving the latest checkpoint
            checkpoint_every (int): Number of steps between saving numbered checkpoints
            max_to_keep (int): Maximum number of numbered checkpoints to keep
            keep_best (bool): Whether to keep the best checkpoint based on a metric
            metric_name (str): Name of the metric to use for best checkpoint selection
            minimize_metric (bool): Whether the metric should be minimized (True) or maximized (False)
            prepare_fn (Callable): Function to prepare state for checkpointing
        """
        self.registry_path = registry_path
        self.run_id = run_id
        self.checkpoint_dir = os.path.join(registry_path, run_id, "checkpoints") if registry_path else None
        self.save_every = save_every
        self.checkpoint_every = checkpoint_every
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.minimize_metric = minimize_metric
        self.best_metric_value = float('inf') if minimize_metric else float('-inf')

        # Function to prepare training state for checkpointing (e.g., moving from devices)
        self.prepare_fn = prepare_fn or (lambda x: jax.device_get(x))

        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _is_better_metric(self, new_value: float) -> bool:
        """
        Check if a new metric value is better than the current best.

        Args:
            new_value (float): New metric value to compare

        Returns:
            bool: True if the new value is better, False otherwise
        """
        if self.minimize_metric:
            return new_value < self.best_metric_value
        else:
            return new_value > self.best_metric_value

    def should_save_latest(self, step: int) -> bool:
        """
        Check if the latest checkpoint should be saved at the current step.

        Args:
            step (int): Current training step

        Returns:
            bool: True if the latest checkpoint should be saved, False otherwise
        """
        return self.checkpoint_dir and step % self.save_every == 0

    def should_save_numbered(self, step: int) -> bool:
        """
        Check if a numbered checkpoint should be saved at the current step.

        Args:
            step (int): Current training step

        Returns:
            bool: True if a numbered checkpoint should be saved, False otherwise
        """
        return self.checkpoint_dir and step % self.checkpoint_every == 0

    def save_checkpoint(
        self,
        train_state: Any,
        step: int = None,
        metrics: Dict[str, float] = None,
        is_best: bool = False,
        suffix: str = None
    ) -> str:
        """
        Save a checkpoint with the given training state.

        Args:
            train_state (Any): Training state to save
            step (int, optional): Current step. If None, extracted from train_state.
            metrics (Dict[str, float], optional): Current metrics to save with checkpoint
            is_best (bool): Whether this is marked as the best checkpoint
            suffix (str, optional): Optional suffix to add to the checkpoint filename

        Returns:
            str: Path to the saved checkpoint, or None if saving failed
        """
        if not self.checkpoint_dir:
            return None

        # Extract step from train_state if not provided
        if step is None and hasattr(train_state, 'step'):
            step = train_state.step

        # Prepare state for saving
        prepared_state = self.prepare_fn(train_state)

        # Determine filename
        if suffix:
            filename = f"state_{suffix}.pyd"
        elif is_best:
            filename = "state_best.pyd"
        else:
            filename = f"state_{step}.pyd"

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        # Save the checkpoint
        try:
            with open(checkpoint_path, "wb") as file:
                checkpoint_data = {
                    'state': prepared_state,
                    'step': step,
                    'time': time.time()
                }

                if metrics:
                    checkpoint_data['metrics'] = metrics

                pickle.dump(checkpoint_data, file)
            return checkpoint_path
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None

    def save_latest(self, train_state: Any, step: int = None, metrics: Dict[str, float] = None) -> str:
        """
        Save the latest checkpoint.

        Args:
            train_state (Any): Training state to save
            step (int, optional): Current step
            metrics (Dict[str, float], optional): Current metrics

        Returns:
            str: Path to the saved checkpoint, or None if saving failed
        """
        return self.save_checkpoint(train_state, step, metrics, suffix="latest")

    def update_best_checkpoint(self, train_state: Any, metric_value: float, step: int = None) -> bool:
        """
        Update the best checkpoint if the given metric value is better.

        Args:
            train_state (Any): Training state to save
            metric_value (float): Value of the metric to compare
            step (int, optional): Current step

        Returns:
            bool: True if a new best checkpoint was saved, False otherwise
        """
        if not self.keep_best:
            return False

        if self._is_better_metric(metric_value):
            self.best_metric_value = metric_value
            metrics = {self.metric_name: metric_value}
            self.save_checkpoint(train_state, step, metrics, is_best=True)
            return True
        return False

    def maybe_save_checkpoints(
        self,
        train_state: Any,
        step: int,
        metrics: Dict[str, float] = None
    ) -> Dict[str, str]:
        """
        Save checkpoints if appropriate based on current step and configuration.

        This method handles saving both latest and numbered checkpoints according
        to the configured intervals, and also updates the best checkpoint if needed.

        Args:
            train_state (Any): Training state to save
            step (int): Current training step
            metrics (Dict[str, float], optional): Current metrics

        Returns:
            Dict[str, str]: Paths to saved checkpoints by type
        """
        saved_paths = {}

        # Save latest checkpoint if it's time
        if self.should_save_latest(step):
            path = self.save_latest(train_state, step, metrics)
            if path:
                saved_paths['latest'] = path

        # Save numbered checkpoint if it's time
        if self.should_save_numbered(step):
            path = self.save_checkpoint(train_state, step, metrics)
            if path:
                saved_paths['numbered'] = path
                # Clean up old checkpoints if needed
                self._cleanup_old_checkpoints()

        # Update best checkpoint if appropriate
        if self.keep_best and metrics and self.metric_name in metrics:
            if self.update_best_checkpoint(train_state, metrics[self.metric_name], step):
                saved_paths['best'] = os.path.join(self.checkpoint_dir, "state_best.pyd")

        return saved_paths

    def _cleanup_old_checkpoints(self):
        """
        Remove old numbered checkpoints to stay within max_to_keep limit.
        """
        if not self.checkpoint_dir or self.max_to_keep is None or self.max_to_keep <= 0:
            return

        # Find all numbered checkpoint files
        files = [f for f in os.listdir(self.checkpoint_dir)
                if f.startswith("state_") and f[6].isdigit() and f.endswith(".pyd")]

        # Parse step numbers
        step_files = []
        for f in files:
            try:
                step = int(f[6:-4])
                step_files.append((step, f))
            except ValueError:
                continue

        # Sort by step number (descending)
        step_files.sort(reverse=True)

        # Remove files beyond the max_to_keep limit
        for _, filename in step_files[self.max_to_keep:]:
            try:
                os.remove(os.path.join(self.checkpoint_dir, filename))
            except OSError:
                pass


    def load_checkpoint(self, path_or_index: Union[str, int]) -> Dict[str, Any]:
        if not self.checkpoint_dir:
            return None

        if path_or_index == -1:
            path = os.path.join(self.checkpoint_dir, "state_latest.pyd")
        else:
            path = os.path.join(self.checkpoint_dir, f"state_{path_or_index}.pyd")

        if not os.path.exists(path):
            return None

        try:
            with open(path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}")
            return None

    def load_latest(self) -> Dict[str, Any]:
        """
        Load the latest checkpoint.

        Returns:
            Dict[str, Any]: Loaded checkpoint data, or None if loading failed
        """
        if not self.checkpoint_dir:
            return None
        return self.load_checkpoint(-1)

    def load_best(self) -> Dict[str, Any]:
        """
        Load the best checkpoint.

        Returns:
            Dict[str, Any]: Loaded checkpoint data, or None if loading failed
        """
        if not self.checkpoint_dir or not self.keep_best:
            return None

        path = os.path.join(self.checkpoint_dir, "state_best.pyd")
        if not os.path.exists(path):
            return None

        return self.load_checkpoint(path)

    def list_available_checkpoints(self) -> Dict[str, str]:
        """
        List all available checkpoints.

        Returns:
            Dict[str, str]: Dictionary of available checkpoint types and their paths
        """
        if not self.checkpoint_dir:
            return {}

        checkpoints = {}

        # Check for latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "state_latest.pyd")
        if os.path.exists(latest_path):
            checkpoints['latest'] = latest_path

        # Check for best checkpoint
        best_path = os.path.join(self.checkpoint_dir, "state_best.pyd")
        if os.path.exists(best_path):
            checkpoints['best'] = best_path

        # List all numbered checkpoints
        numbered = {}
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("state_") and filename[6].isdigit() and filename.endswith(".pyd"):
                try:
                    step = int(filename[6:-4])
                    numbered[step] = os.path.join(self.checkpoint_dir, filename)
                except ValueError:
                    continue

        if numbered:
            # Add the most recent checkpoint
            latest_step = max(numbered.keys())
            checkpoints[f"step_{latest_step}"] = numbered[latest_step]
            checkpoints["numbered"] = numbered

        return checkpoints
