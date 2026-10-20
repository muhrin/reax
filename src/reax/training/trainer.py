from collections.abc import Callable, Generator, Sequence
import contextlib
import datetime
import logging
import os
import signal
import sys
from typing import TYPE_CHECKING, Any, Final, TypeVar
import weakref

import beartype
from flax import nnx
import jaxtyping as jt
from lightning_utilities.core import rank_zero
from typing_extensions import override

from .. import _engine as engine_
from .. import data, exceptions, hooks, keys, stages, strategies
from .. import listeners as listeners_
from .. import loggers as loggers_
from ..utils import events
from . import _checkpointing, _hooks_map, _logger_connector, _utils

if TYPE_CHECKING:
    from collections.abc import Iterable

    import reax

_StageT = TypeVar("_StageT", bound=stages.Stage)
_LoopT = TypeVar("_LoopT", bound=stages.Loop)

_LOGGER = logging.getLogger(__name__)

__all__ = ("Trainer",)


class Trainer(stages.StageListener):
    # pylint: disable=too-many-public-methods

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        *,
        accelerator: str = "auto",
        strategy: "str | reax.Strategy" = "auto",
        devices: list[int] | str | int = "auto",
        precision=None,
        logger: "reax.Logger | Iterable[reax.Logger] | bool | None" = True,
        listeners: "list[reax.TrainerListener] | reax.TrainerListener | None" = None,
        log_every_n_steps: int = 50,
        enable_checkpointing: bool | None = True,
        enable_progress_bar: bool = True,
        enable_model_summary: bool | None = None,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
        default_root_dir: "reax.types.Path | None" = None,
        checkpointing: "reax.Checkpointing | None" = None,
        profiler: "reax.Profiler | None" = None,
        metric_evaluator: "reax.metrics.MetricEvaluator | None" = None,
    ):
        """Create a Trainer.

        Args:
            accelerator: Hardware accelerator to use (e.g. ``"auto"``, ``"cpu"``, ``"gpu"``).
            strategy: Parallelism strategy to use (e.g. ``"auto"`` or a
                :class:`reax.Strategy` instance).
            devices: Number or list of devices to use (e.g. ``"auto"``, ``1``,
                ``[0, 1]``).
            precision: Numerical precision to use for training (e.g. ``"32"``).
            logger: Logging backend configuration. ``True`` uses the default CSV logger,
                ``False`` disables logging, or a :class:`reax.Logger` / list of loggers.
            listeners: A :class:`reax.TrainerListener` or list of listeners to register
                with the trainer for stage/lifecycle hooks.
            log_every_n_steps: Number of training steps between metric logging calls.
            enable_checkpointing: Whether to auto-configure a default checkpointing
                listener when none is provided in ``listeners``.
            enable_progress_bar: Whether to display a progress bar during stage execution.
            enable_model_summary: Whether to print a model parameter summary at start.
                Not yet supported; passing ``True`` emits a warning.
            deterministic: If ``True``, enable deterministic training (reproducible
                results, potentially slower).
            rngs: Optional :class:`flax.nnx.Rngs` instance for controlling random
                number generation. If ``None``, the engine's default RNGs are used.
            default_root_dir: Default directory path used as the base for loggers,
                checkpoints, and other artefacts. If ``None``, the current working
                directory is used.
            checkpointing: A custom :class:`reax.Checkpointing` implementation.
                If ``None``, a default :class:`MsgpackCheckpointing` is used.
            profiler: An optional :class:`reax.Profiler` for performance profiling
                of stage execution.
            metric_evaluator: An optional :class:`reax.metrics.MetricEvaluator` used
                when evaluating metrics during stages.
        """
        # Params
        self._log_every_n_steps: Final[int] = log_every_n_steps
        self._default_root_dir = (
            os.fspath(default_root_dir) if default_root_dir is not None else os.getcwd()
        )

        self._automatic_optimization = True
        self._optimizers = []
        self._stage: stages.Stage | None = None

        # State
        self._engine = engine_.Engine(
            accelerator,
            strategy,
            devices,
            precision=precision,
            logger=logger,
            listeners=listeners,
            deterministic=deterministic,
            rngs=rngs,
            default_root_dir=default_root_dir,
            profiler=profiler,
            metric_evaluator=metric_evaluator,
        )

        self._current_epoch: int = -1
        self._epochs_completed: int = 0
        self._global_updates: int = 0
        self._checkpointing = (
            checkpointing if checkpointing is not None else _checkpointing.MsgpackCheckpointing()
        )

        # Init
        self._events = events.EventGenerator[hooks.TrainerListener](
            default_args=(weakref.proxy(self),)
        )

        self._logging = _logger_connector.TrainerLogging()

        if isinstance(listeners, hooks.TrainerListener):
            listeners = [listeners]
        if listeners:
            for listener in listeners:
                self._events.add_listener(listener)

        pbar = _utils.init_progress_bar(self._events.find(), enable_progress_bar)
        if pbar is not None:
            self._events.add_listener(pbar)

        if enable_model_summary:
            _LOGGER.warning("`enable_model_summary` is not supported yet, ignoring")

        if self.checkpoint_listeners:
            if not enable_checkpointing:
                raise exceptions.MisconfigurationException(
                    "Trainer was configured with `enable_checkpointing=False`"
                    " but found `ModelCheckpoint` in listeners list."
                )
        elif enable_checkpointing:
            # Create the default checkpointer
            self._events.add_listener(listeners_.ModelCheckpoint())

    def finalize(self):
        """Clean up the trainer.

        After this called this object should no longer be interacted with.
        """
        self._engine.finalize()
        self._events.reset()

    def __del__(self):
        """Del function."""
        self.finalize()

    @property
    def engine(self) -> "reax.Engine":
        """The trainer's engine"""
        return self._engine

    @property
    def strategy(self) -> strategies.Strategy:
        """Strategy function."""
        return self._engine.strategy

    @property
    def rngs(self) -> nnx.Rngs:
        return self._engine.rngs

    @property
    def profiler(self) -> "reax.Profiler":
        return self._engine.profiler

    @property
    def is_global_zero(self) -> bool:
        """Whether this process is the global zero in multi-node training.

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                if self.trainer.is_global_zero:
                    print("in node 0, accelerator 0")
        """
        return self.engine.is_global_zero

    @property
    def current_epoch(self) -> int:
        """Get the current fitting epoch."""
        return self._current_epoch

    @property
    def epochs_completed(self) -> int:
        """Get the number of training epochs that have completed.

        Unlike :attr:`current_epoch`, which is the (zero-based) index of the epoch in progress,
        this is a count, and the epoch in progress is only included once it has finished.
        """
        return self._epochs_completed

    @property
    def global_updates(self) -> int:
        """Get the global number of optimizer updates."""
        if self._stage is not None and isinstance(self._stage, stages.Fit):
            # The stage is running, so count add the number of updates it has performed so far
            return self._global_updates + self._stage.global_updates

        return self._global_updates

    @property
    def optimizers(self) -> list["reax.Optimizer"]:
        """Optimizers function."""
        return self._optimizers

    @optimizers.setter
    def optimizers(self, opts: list["reax.Optimizer"]) -> None:
        """Optimizers function."""
        self._optimizers = opts

    @property
    def metric_evaluator(self) -> "reax.metrics.MetricEvaluator":
        """Get the metric evaluator."""
        return self._engine.metric_evaluator

    @property
    def should_stop(self) -> bool:
        """Should stop."""
        if self._stage is None:
            return False
        return self._stage.should_stop

    @should_stop.setter
    def should_stop(self, stop: bool):
        """Should stop."""
        if stop and self._stage is not None:
            self._stage.stop("None")

    @property
    def sanity_checking(self) -> bool:
        """`True` if currently sanity checking, `False` otherwise"""
        return isinstance(self._stage, stages.Fit) and self._stage.sanity_checking

    @property
    def default_root_dir(self) -> str:
        """Get the fallback directory used for loggers and other components when not explicitly
        specified.
        """
        if _utils.is_local_file_protocol(self._default_root_dir):
            return os.path.normpath(os.path.expanduser(self._default_root_dir))

        return self._default_root_dir

    @property
    def checkpointing(self) -> "reax.Checkpointing":
        return self._checkpointing

    # region Listeners

    @property
    def listeners(self) -> "list[reax.TrainerListener]":
        return self._events.find()

    @property
    def early_stopping_listener(self) -> listeners_.EarlyStopping | None:
        """The first :class:`~reax.listeners.early_stopping.EarlyStopping` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist.
        """
        listeners = self.early_stopping_listeners
        return listeners[0] if len(listeners) > 0 else None

    @property
    def early_stopping_listeners(self) -> list[listeners_.EarlyStopping]:
        """A list of all instances of :class:`~reax.listeners.early_stopping.EarlyStopping` found in
        the Trainer.listeners list.
        """
        return self._events.find(type=listeners_.EarlyStopping)

    @property
    def checkpoint_listeners(self) -> list["reax.listeners.Checkpointer"]:
        """The first :class:`~reax.listeners.model_checkpoint.ModelCheckpoint` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist.
        """
        return self._events.find(type=listeners_.Checkpointer)

    @property
    def checkpoint_listener(self) -> "reax.listeners.Checkpointer | None":
        """The first :class:`~reax.listeners.model_checkpoint.ModelCheckpoint` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist.
        """
        listeners = self.checkpoint_listeners
        return listeners[0] if len(listeners) > 0 else None

    @property
    def progress_bar_listeners(self) -> list["reax.listeners.ProgressBar"]:
        """The first :class:`~reax.listeners.ProgressBar` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist.
        """
        return self._events.find(type=listeners_.ProgressBar)

    @property
    def progress_bar_listener(self) -> "reax.listeners.ProgressBar | None":
        """The first :class:`~reax.listeners.ProgressBar` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist.
        """
        listeners = self.progress_bar_listeners
        return listeners[0] if len(listeners) > 0 else None

    # end region

    @property
    def logger(self) -> "reax.Logger | None":
        r"""Get the first (and main) logger."""
        return self._engine.logger

    @property
    def loggers(self) -> list["reax.Logger"]:
        """Get all the loggers."""
        return self._engine.loggers

    @loggers.setter
    def loggers(self, loggers: list["reax.Logger"] | None) -> None:
        """Loggers function."""
        self._engine.loggers = loggers

    @property
    def log_dir(self) -> str:
        """The directory for the current experiment. Use this to save images to, etc...

        Note:
            You must call this on all processes. Failing to do so will cause your program to
            stall forever.

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                img = ...
                save_img(img, self.trainer.log_dir)
        """
        if len(self.loggers) > 0:
            if not isinstance(self.loggers[0], loggers_.TensorBoardLogger):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.engine.broadcast(dirpath)
        if dirpath is None:
            dirpath = "."

        return dirpath

    @property
    def stage(self) -> "reax.Stage | None":
        """Get the current stage if there is one running, otherwise None."""
        return self._stage

    @property
    def progress_bar_metrics(self) -> dict:
        """Progress bar metrics."""
        return self._logging.progress_bar_metrics

    @property
    def listener_metrics(self) -> dict:
        """Callback metrics."""
        return self._logging.listener_metrics

    @property
    def logged_metrics(self) -> dict:
        """Logger metrics."""
        return self._logging.logged_metrics

    @property
    def process_index(self) -> int:
        """Global rank."""
        return self._engine.process_index

    @property
    def local_process_index(self) -> int:
        """Local rank."""
        return self._engine.local_process_index

    @property
    def node_rank(self) -> int:
        """Node rank."""
        return self._engine.node_rank

    @property
    def process_count(self) -> int:
        """World size."""
        return self._engine.process_count

    @property
    def train_dataloader(self) -> "reax.data.DataLoader | None":
        """Train dataloader."""
        if self._stage is not None:
            return getattr(self._stage, "train_dataloader", None)
        return None

    @property
    def val_dataloaders(self) -> "reax.data.DataLoader | None":
        """Train dataloader."""
        if self._stage is not None:
            return getattr(self._stage, "val_dataloader", None)
        return None

    @property
    def _module(self) -> "reax.Module":
        if self._stage is None:
            raise AttributeError("There is no stage running, so module is not available")
        if not isinstance(self._stage, stages.ModuleStage):
            raise AttributeError(
                f"The running stage '{self._stage.__class__.__name__}', does not have a module"
            )
        return self._stage.module

    def log(
        # pylint: disable=unused-argument
        self,
        name: str,
        value,
        *,
        prog_bar: bool = False,
        batch_size: int | None = None,
        logger: bool | None = None,
        on_step=True,
        on_epoch=True,
        reduce_fn: "reax.types.ReduceFn" = "mean",
    ) -> None:
        """Log a metric value during a stage.

        Args:
            name: Name of the metric.
            value: The scalar or array value to log.
            prog_bar: If ``True``, the metric is also shown on the progress bar.
            batch_size: Number of samples in the batch. If ``None``, it is inferred
                from the batch.
            logger: If ``True``, the metric is written to the configured logger(s).
                If ``False``, it is only made available to listeners.
            on_step: If ``True``, the metric is logged at every batch (step).
            on_epoch: If ``True``, the metric is aggregated and logged once at the
                end of the epoch.
            reduce_fn: Aggregation function used to combine per-step values when
                logging on epoch (e.g. ``"mean"``, ``"sum"``).

        Raises:
            RuntimeError: If no stage is currently running.
        """
        if self._stage is None:
            raise RuntimeError(
                "Logging is only supported during one of the train/validate/test stages. "
                "There is currently no stage running."
            )

        # Let the stage take care of it. ``None`` means "not specified", which
        # downstream is treated the same as ``False``.
        self._stage.log(
            name,
            value,
            prog_bar=prog_bar,
            logger=logger if logger is not None else False,
            batch_size=batch_size,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fn=reduce_fn,
        )

    # region Stages
    @jt.jaxtyped(typechecker=beartype.beartype)
    def fit(
        self,
        module: "reax.Module",
        train_dataloaders: "reax.DataLoader | None" = None,
        val_dataloaders: "reax.DataLoader | None" = None,
        *,
        datamodule: "reax.DataModule | None" = None,
        ckpt_path: "reax.types.Path | None" = None,
        fast_dev_run: bool | int = False,
        max_epochs: int | None = 1_000,
        min_epochs: int = 0,
        min_updates: int = 0,
        max_updates: int | float | None = None,
        max_time: str | datetime.timedelta | dict[str, int] | None = None,
        limit_train_batches: int | float | None = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: int | float | None = 1.0,
        val_check_interval: int | float = 1.0,
        check_val_every_n_epoch: int = 1,
        num_sanity_val_steps: int | None = 2,
        reload_dataloaders_every_n_epochs: int = 0,
    ) -> "reax.stages.Fit":
        """Train the module.

        Args:
            module: The :class:`reax.Module` to train.
            train_dataloaders: A :class:`reax.DataLoader` for training data.
                If ``None``, the dataloader is taken from ``datamodule``.
            val_dataloaders: A :class:`reax.DataLoader` for validation data.
                If ``None``, the dataloader is taken from ``datamodule``.
            datamodule: A :class:`reax.DataModule` that provides all dataloaders.
            ckpt_path: Path/URL of the checkpoint from which training is resumed.
                Could also be one of two special keywords ``"last"`` and ``"hpc"``.
                If there is no checkpoint file at the path, an exception is raised.
            fast_dev_run: If ``True``, run only 1 batch of train and val steps for a
                fast (unit) test. Can be set to an integer to run N batches.
            max_epochs: Maximum number of epochs to train for. If ``0``, no
                limit is applied. Defaults to 1000.
            min_epochs: Minimum number of epochs to train for. If non-zero, training
                will not stop before this many epochs are complete.
            min_updates: Minimum number of optimizer updates to perform, regardless
                of epoch count.
            max_updates: Maximum number of optimizer updates to perform. If ``None``,
                no limit based on update count is applied.
            max_time: Stop after this wall-clock duration. Accepts a string in
                ``"DD:HH:MM:SS"`` format, a :class:`datetime.timedelta`, or a
                ``dict[str, int]`` passed to ``timedelta(**kwargs)``.
            limit_train_batches: Maximum number of training batches per epoch. If
                ``int``, limits the absolute batch count. If ``float`` in [0, 1],
                limits a fraction of the available batches. ``1.0`` (default) uses all.
            accumulate_grad_batches: Number of batches to accumulate before performing
                one optimizer step. Defaults to 1.
            limit_val_batches: Maximum number of validation batches per epoch. Uses
                the same semantics as ``limit_train_batches``.
            val_check_interval: How often to run validation within a training epoch.
                If ``float`` in (0, 1), runs after that fraction of the epoch. If
                ``int``, runs after that many batches.
            check_val_every_n_epoch: Run validation only once every N epochs. If
                ``None``, the check is decoupled from the epoch boundary and
                ``val_check_interval`` is used globally.
            num_sanity_val_steps: Number of validation steps to run as a sanity check
                before the actual training starts. Defaults to ``2``. Set to ``0`` to
                disable the sanity check.
            reload_dataloaders_every_n_epochs: Reload dataloaders from scratch every
                N epochs. ``0`` (default) means no reload.

        Returns:
            The completed :class:`reax.stages.Fit` stage.
        """
        if max_updates is not None and max_updates < 0:
            raise ValueError("`max_updates` must be a non-negative integer")
        if max_epochs is not None and max_epochs < 0:
            raise ValueError(f"`max_epochs` must be a non-negative integer or {keys.NO_LIMIT}")

        if fast_dev_run:
            num_batches = 1 if fast_dev_run is True else fast_dev_run
            rank_zero.rank_zero_info(
                f"Running in `fast_dev_run` mode: will run the requested loop using {num_batches} "
                f"batch(es). Logging and checkpointing is suppressed."
            )

        if ckpt_path:
            self._load_module_params(module, ckpt_path)

        datamanager = self._create_data_manager(
            module=module,
            datamodule=datamodule,
            train=train_dataloaders,
            val=val_dataloaders,
        )
        fit = stages.Fit(
            module,
            datamanager,
            self.optimizers,
            self.engine,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            min_updates=min_updates,
            max_updates=max_updates,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_val_batches=limit_val_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        )
        self._run_stage(fit)
        # Update state variables
        self._global_updates += fit.global_updates

        return fit

    @jt.jaxtyped(typechecker=beartype.beartype)
    def validate(
        self,
        module: "reax.Module",
        dataloaders=None,
        datamodule=None,
        ckpt_path: "reax.types.Path | None" = None,
        limit_batches: int | float | None = keys.NO_LIMIT,
    ) -> "reax.stages.Validate":
        """Run validation on the provided module.

        Args:
            module: The :class:`reax.Module` to evaluate.
            dataloaders: A :class:`reax.DataLoader` of validation data. If ``None``,
                the dataloader is obtained from ``datamodule``.
            datamodule: A :class:`reax.DataModule` providing the validation dataloader.
            ckpt_path: Path/URL of the checkpoint to load before validation. If set,
                the module is loaded from this checkpoint.
            limit_batches: Maximum number of validation batches to process. If ``int``,
                limits the absolute count. If ``float`` in (0, 1], limits a fraction.
                Defaults to :attr:`reax.keys.NO_LIMIT` (all batches).

        Returns:
            The completed :class:`reax.stages.Validate` stage.
        """
        if ckpt_path:
            self._load_module_params(module, ckpt_path)

        datamanager = self._create_data_manager(module, datamodule, val=dataloaders)
        validate = stages.Validate(module, datamanager, self.engine, limit_batches=limit_batches)
        self._run_stage(validate)

        return validate

    @jt.jaxtyped(typechecker=beartype.beartype)
    def test(
        self,
        module: "reax.Module",
        dataloaders=None,
        datamodule=None,
        ckpt_path: "reax.types.Path | None" = None,
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = 1.0,
    ) -> "reax.stages.Test":
        """Run the test loop on the provided module.

        Args:
            module: The :class:`reax.Module` to test.
            dataloaders: A :class:`reax.DataLoader` of test data. If ``None``, the
                dataloader is obtained from ``datamodule``.
            datamodule: A :class:`reax.DataModule` providing the test dataloader.
            ckpt_path: Path/URL of the checkpoint to load before testing. If set, the
                module is loaded from this checkpoint.
            fast_dev_run: If ``True``, run only 1 batch for a fast (unit) test.
                Can be set to an integer to run N batches.
            limit_batches: Maximum number of test batches to process. If ``int``,
                limits the absolute count. If ``float`` in (0, 1], limits a fraction.
                ``1.0`` (default) uses all batches.

        Returns:
            The completed :class:`reax.stages.Test` stage.
        """
        if ckpt_path:
            self._load_module_params(module, ckpt_path)

        datamanager = self._create_data_manager(module, datamodule, test=dataloaders)
        test = stages.Test(
            module,
            datamanager,
            self.engine,
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
        )
        self._run_stage(test)

        return test

    @jt.jaxtyped(typechecker=beartype.beartype)
    def predict(
        self,
        module: "reax.Module",
        dataloaders: "reax.DataLoader | None" = None,
        datamodule: "reax.DataModule | None" = None,
        ckpt_path: "reax.types.Path | None" = None,
        keep_predictions: bool | None = None,
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = keys.NO_LIMIT,
    ) -> "reax.stages.Predict":
        r"""Run inference on the data.

        Logging is disabled in the predict hooks.

        Args:
            module: The :class:`reax.Module` to run predictions with.
            dataloaders: A :class:`reax.DataLoader` of data to run predictions on.
                If ``None``, the dataloader is obtained from ``datamodule``.
            datamodule: A :class:`reax.DataModule` providing the predict dataloader.
            ckpt_path: Path/URL of the checkpoint to load before prediction. If set,
                the module is loaded from this checkpoint.
            keep_predictions: If ``True``, store all batch outputs on the stage's
                ``predictions`` attribute after the run. If ``None`` (default) this
                is enabled; set to ``False`` to discard outputs.
            fast_dev_run: If ``True``, run only 1 batch for a fast (unit) test.
                Can be set to an integer to run N batches.
            limit_batches: Maximum number of batches to process. If ``int``, limits
                the absolute count. If ``float`` in (0, 1], limits a fraction.
                Defaults to :attr:`reax.keys.NO_LIMIT` (all batches).

        Returns:
            The completed :class:`reax.stages.Predict` stage.
        """
        if ckpt_path:
            self._load_module_params(module, ckpt_path)

        if fast_dev_run:
            limit_batches = 1

        if keep_predictions is None:
            keep_predictions = True

        datamanager = self._create_data_manager(module, datamodule, predict=dataloaders)
        predict = stages.Predict(
            module,
            datamanager,
            self.engine,
            limit_batches=limit_batches,
            keep_predictions=keep_predictions,
        )
        self._run_stage(predict)

        return predict

    @jt.jaxtyped(typechecker=beartype.beartype)
    def eval_stats(
        self,
        stats: "reax.Metric | Sequence[reax.Metric] | dict[str, reax.Metric]",
        dataloaders: "reax.DataLoader | None" = None,
        datamodule: "reax.DataModule | None" = None,
        dataset_name: str = "train",
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = keys.NO_LIMIT,
        metric_evaluator: "reax.metrics.MetricEvaluator | None" = None,
    ) -> "reax.stages.EvaluateStats":
        r"""Evaluate metrics on a dataset to get statistics about it.

        The calculated statistics will be stored on `stage.logged_metrics` where `stage` is the
        return value of this call.

        Args:
            stats: A single :class:`reax.Metric`, a :class:`Sequence` of metrics, or a
                ``dict[str, reax.Metric]`` mapping names to metrics to evaluate.
            dataloaders: A :class:`reax.DataLoader` providing the data to evaluate. If
                ``None``, the dataloader is obtained from ``datamodule``.
            datamodule: A :class:`reax.DataModule` providing the dataloader.
            dataset_name: The name of the data loader to fetch from the datamodule
                (e.g. ``"train"``, ``"val"``). Defaults to ``"train"``.
            fast_dev_run: If ``True``, run only 1 batch for a fast (unit) test.
                Can be set to an integer to run N batches.
            limit_batches: Maximum number of batches to process. If ``int``, limits
                the absolute count. If ``float`` in (0, 1], limits a fraction.
                Defaults to :attr:`reax.keys.NO_LIMIT` (all batches).
            metric_evaluator: Optional :class:`reax.metrics.MetricEvaluator` to use.
                If ``None``, falls back to the trainer's engine-level evaluator.

        Returns:
            The completed :class:`reax.stages.EvaluateStats` stage.
        """
        if fast_dev_run:
            limit_batches = 1

        # ``dataset_name`` is dynamic, so the loader is passed via ``**{name: loader}``.
        # The checker cannot resolve a dynamically-keyed kwarg and conservatively
        # checks it against every parameter (including ``module``), but the loader
        # is never bound to ``module`` at runtime.
        datamanager = self._create_data_manager(
            datamodule=datamodule,
            **{dataset_name: dataloaders},  # ty: ignore[invalid-argument-type]
        )

        evaluator = (
            metric_evaluator if metric_evaluator is not None else self.engine.metric_evaluator
        )

        eval_stats = stages.EvaluateStats(
            stats,
            datamanager,
            self.engine,
            dataset_name=dataset_name,
            limit_batches=limit_batches,
            evaluator=evaluator,
        )
        self._run_stage(eval_stats)

        return eval_stats

    def run(self, stage: _StageT) -> _StageT:
        return self._run_stage(stage)

    def _run_stage(self, stage: _StageT) -> _StageT:
        """Run stage."""
        with self.engine.default_device():
            try:
                with self._attach(stage), self.engine.profile(stage.name):
                    with stage.events.listen_context(self):
                        stage.run()
            except KeyboardInterrupt as exc:
                rank_zero.rank_zero_info(
                    "\nDetected KeyboardInterrupt, attempting graceful shutdown ..."
                )
                # Disable further Ctrl+C presses while we respond to this one
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                self._interrupt(stage, exc)
                sys.exit(1)
            except BaseException as exc:
                self._interrupt(stage, exc)
                raise

            else:
                for logger in self.loggers:
                    logger.finalize("success")

        return stage

    def _interrupt(self, stage: "reax.Stage", exception: BaseException):
        # Inform our listeners of the exception
        self._events.fire_event(hooks.TrainerListener.on_exception, stage, exception)

        # TODO: self._strategy.on_exception(exc)
        for logger in self.loggers:
            logger.finalize("failed")

    # endregion

    @contextlib.contextmanager
    def _attach(self, stage: stages.Stage):
        """Attach function."""
        self._stage = stage
        # TODO: Analyse this
        if isinstance(stage, stages.ModuleStage):
            listeners = stage.module.configure_listeners()
            module_trainer_ctx = stage.module.attach
        else:
            listeners = []
            module_trainer_ctx = contextlib.nullcontext

        try:
            with module_trainer_ctx(weakref.proxy(self)), self._attach_model_listeners(listeners):
                yield
        finally:
            self._stage = None

    @contextlib.contextmanager
    def _attach_model_listeners(
        self, listeners: "reax.TrainerListener | Sequence[reax.TrainerListener]"
    ) -> Generator[None, Any, None]:
        """Attaches the listeners defined in the model.

        If a listener returned by the model's configure_listener method has the same type as one or
        several listeners already present in the trainer listeners list, it will replace them.
        In addition, all :class:`~lightning.pytorch.listeners.model_checkpoint.ModelCheckpoint`
        listeners will be pushed to the end of the list, ensuring they run last.
        """
        if listeners:
            listeners = [listeners] if not isinstance(listeners, Sequence) else listeners
            model_listener_types = {type(c) for c in listeners}
            trainer_listener_types = {type(c) for c in self.listeners}
            # edge case: if an unmodified listener was added, the logic below would filter it
            trainer_listener_types.discard(hooks.TrainerListener)
            # exclude trainer listeners of the same class or subclass
            override_types = set()
            for model_cb in model_listener_types:
                for trainer_cb in trainer_listener_types:
                    if issubclass(model_cb, trainer_cb):
                        override_types.add(trainer_cb)
                        break
            if override_types:
                rank_zero.rank_zero_info(
                    "The following listeners returned in `Module.configure_listeners` will override"
                    " existing listeners passed to Trainer:"
                    f" {', '.join(sorted(t.__name__ for t in override_types))}"
                )
            # remove all listeners with a type that occurs in model listeners
            all_listeners = [
                listener for listener in self.listeners if type(listener) not in override_types
            ]
            all_listeners.extend(listeners)
            all_listeners = _utils.reorder_listeners(all_listeners)
        else:
            all_listeners = []

        original_events = self._events
        try:
            if listeners:
                self._events = events.EventGenerator[hooks.TrainerListener](
                    default_args=(weakref.proxy(self),)
                )
                for listener in all_listeners:
                    self._events.add_listener(listener)

            yield
        finally:
            self._events = original_events

    @override
    def on_stage_start(self, stage: _StageT, /) -> None:
        """The stage is starting."""
        self._logging.reset_metrics()
        self._events.fire_event(hooks.TrainerListener.setup, stage)

    @override
    def on_stage_started(self, stage: _StageT, /):
        # Fire the generic event, and then the specific
        self._events.fire_event(hooks.TrainerListener.on_stage_start, stage)
        self._fire_stage_event(stage, stages.StageListener.on_stage_started, stage)

    @override
    def on_loop_started(self, stage: _StageT, loop: _LoopT, _max_iters: int | float | None, /):
        if not self._optimizers and isinstance(loop, stages.TrainEpoch):
            self._optimizers = loop.optimizers
        self._events.fire_event(hooks.TrainerListener.on_loop_start, stage, loop)
        self._fire_stage_event(loop, stages.StageListener.on_loop_started, stage, loop)

    def on_loop_iter_starting(self, stage: _StageT, loop: _LoopT, /):
        if isinstance(stage, stages.Fit) and isinstance(loop, stages.FitLoop):
            # Keep track of the number of completed training epochs
            self._current_epoch += 1

    @override
    def on_loop_iter_started(self, stage: _StageT, loop: _LoopT, /):
        """On stage iter started."""
        self._events.fire_event(hooks.TrainerListener.on_loop_iter_start, stage, loop)
        if isinstance(loop, stages.LoopOverBatches):
            self._fire_stage_event(
                loop, stages.StageListener.on_loop_iter_started, stage, loop, loop.batch
            )

    @override
    def on_loop_iter_finished(self, stage: _StageT, loop: _LoopT, outputs: Any, /):
        """On stage iter ending."""
        if isinstance(stage, stages.ModuleStage):
            self._logging.update(stage)
            if stage.logged_metrics:
                logging_metrics = {"epoch": self.current_epoch, "stage": stage.name}
                logging_metrics.update(stage.logged_metrics)
                for logger in self.loggers:
                    logger.log_metrics(metrics=logging_metrics, step=self.global_updates - 1)
                    logger.save()

        self._events.fire_event(hooks.TrainerListener.on_loop_iter_end, stage, loop, outputs)
        if isinstance(loop, stages.LoopOverBatches):
            self._fire_stage_event(
                loop, stages.StageListener.on_loop_iter_finished, stage, loop, outputs
            )

    @override
    def on_loop_finishing(self, stage: _StageT, *_):
        if isinstance(stage, stages.ModuleStage):
            self._logging.update(stage)

            metrics = stage.results
            if metrics[keys.LOG]:
                logging_metrics = {"epoch": self.current_epoch, "stage": stage.name}
                logging_metrics.update(metrics[keys.LOG])
                for logger in self.loggers:
                    logger.log_metrics(metrics=logging_metrics, step=self.global_updates - 1)

    @override
    def on_loop_finished(self, stage: _StageT, loop: _LoopT, /):
        if isinstance(stage, stages.Fit) and isinstance(loop, stages.TrainEpoch):
            # Count the epoch before firing the hooks, so that `on_train_epoch_end` sees it
            self._epochs_completed += 1

        self._events.fire_event(hooks.TrainerListener.on_loop_end, stage, loop, None)
        self._fire_stage_event(loop, stages.StageListener.on_loop_finished, stage, loop)

    @override
    def on_stage_finished(self, stage: _StageT, /):
        self._events.fire_event(hooks.TrainerListener.on_stage_end, stage)
        self._fire_stage_event(stage, stages.StageListener.on_stage_finished, stage)
        # Finally, call teardown
        self._events.fire_event(hooks.TrainerListener.teardown, stage)

    # region Checkpointing

    def save_checkpoint(self, filepath: "reax.types.Path", weights_only: bool = True):
        """For now, we just save the model weights.

        The user has to store the model definition themselves.

        Args:
            filepath: Destination path (local or remote URL) for the checkpoint file.
            weights_only: If ``True``, only the module's parameter values are
                serialised. If ``False``, additional state is included (not yet
                implemented).
        """
        ckpt = _checkpointing.dump_checkpoint(self._module, weights_only=weights_only)
        self._checkpointing.save(ckpt, filepath)

    def _get_checkpoint_path(self) -> str | None:
        """Get checkpoint path."""
        checkpointers = self.checkpoint_listeners
        if not checkpointers:
            return None

        if len(checkpointers) > 1:
            fn = self._get_checkpoint_path
            rank_zero.rank_zero_warn(
                f'`.{fn}(ckpt_path="best")` found Trainer with multiple `ModelCheckpoint`'
                " listeners. The best checkpoint path from first checkpoint listener will be used."
            )

        checkpointer = checkpointers[0]
        return getattr(checkpointer, "best_model_path", None)

    # endregion

    def _fire_stage_event(self, sender: stages.Stage | _LoopT, hook: Callable, *args):
        event = _hooks_map.hook_map(sender).get(hook)
        if event is not None:
            self._events.fire_event(event, *args)

    def _load_module_params(self, module: "reax.Module", filename: "reax.types.Path"):
        ckpt = self._checkpointing.load(filename)
        module.set_parameters(ckpt[_checkpointing.PARAMS])

    def _create_data_manager(
        self,
        module: "reax.data.DataSource | None" = None,
        datamodule: "reax.data.DataModule | None" = None,
        **loaders,
    ) -> data.DataSourceManager:
        return data.create_manager(self._engine, module=module, datamodule=datamodule, **loaders)
