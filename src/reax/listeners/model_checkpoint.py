import os
import re
from typing import TYPE_CHECKING, Literal, Optional

import jax
from typing_extensions import override

from reax import TrainerListener, stages

if TYPE_CHECKING:
    import reax


class ModelCheckpoint(TrainerListener):

    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_EQUALS_CHAR = "="
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"
    STARTING_VERSION = 1

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        mode: Literal["min", "max"] = "min",
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        enable_version_counter: bool = True,
    ):
        super().__init__()
        self._monitor = monitor
        self._mode = mode
        self._every_n_epochs = every_n_epochs
        self._save_on_train_epoch_end = save_on_train_epoch_end
        self._enable_version_counter = enable_version_counter
        self.save_last = save_last
        self._save_top_k = save_top_k
        self.dirpath = os.path.realpath(os.path.expanduser(dirpath)) or dirpath
        self.filename = filename

        # State
        self.last_model_path = ""
        self._last_checkpoint_saved = ""

    @override
    def on_epoch_ending(
        self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage", metrics: dict
    ) -> None:
        """Save a checkpoint at the end of the training epoch."""
        if not self._should_skip_saving_checkpoint(
            trainer
        ) and self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = stage.callback_metrics
            if (
                self._every_n_epochs >= 1
                and (trainer.current_epoch + 1) % self._every_n_epochs == 0
            ):
                self._save_topk_checkpoint(trainer, monitor_candidates)

            self._save_last_checkpoint(trainer, monitor_candidates)

    def _should_skip_saving_checkpoint(self, trainer: "reax.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return (
            # bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            not isinstance(trainer.stage, stages.Fit)  # don't save anything during non-fit
            # or trainer.sanity_checking  # don't save anything during sanity check
            # or self._last_global_step_saved == trainer.global_step  # already saved at the last step
        )

    def _should_save_on_train_epoch_end(self, trainer: "reax.Trainer") -> bool:
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end

        # if `check_val_every_n_epoch != 1`, we can't say when the validation dataloader will be loaded
        # so let's not enforce saving at every training epoch end
        if trainer.check_val_every_n_epoch != 1:
            return False

        # no validation means save on train epoch end
        num_val_batches = (
            sum(trainer.num_val_batches)
            if isinstance(trainer.num_val_batches, list)
            else trainer.num_val_batches
        )
        if num_val_batches == 0:
            return True

        # if the user runs validation multiple times per training epoch, then we run after validation
        # instead of on train epoch end
        # return trainer.val_check_interval == 1.0
        return True

    def _save_last_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: dict[str, jax.Array]
    ) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while os.path.exists(filepath) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(
                    monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt
                )
                version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        if self.save_last == "link" and self._last_checkpoint_saved and self._save_top_k != 0:
            self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
        else:
            self._save_checkpoint(trainer, filepath)

        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(previous)

    def format_checkpoint_name(
        self,
        metrics: dict[str, jax.Array],
        filename: Optional[str] = None,
        ver: Optional[int] = None,
    ) -> str:
        """Generate a filename according to the defined template.

        Example::

            >>> tmpdir = os.path.dirname(__file__)
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=0)))
            'epoch=0.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch:03d}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=5)))
            'epoch=005.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}-{val_loss:.2f}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
            'epoch=2-val_loss=0.12.ckpt'
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.12), filename='{epoch:d}'))
            'epoch=2.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir,
            ... filename='epoch={epoch}-validation_loss={val_loss:.2f}',
            ... auto_insert_metric_name=False)
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
            'epoch=2-validation_loss=0.12.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{missing:d}')
            >>> os.path.basename(ckpt.format_checkpoint_name({}))
            'missing=0.ckpt'
            >>> ckpt = ModelCheckpoint(filename='{step}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(step=0)))
            'step=0.ckpt'

        """
        filename = filename or self.filename
        filename = self._format_checkpoint_name(
            filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name
        )

        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name

    def _format_checkpoint_name(
        self,
        filename: Optional[str],
        metrics: dict[str, jax.Array],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + self.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)

        # sort keys from longest to shortest to avoid replacing substring
        # eg: if keys are "epoch" and "epoch_test", the latter must be replaced first
        groups = sorted(groups, key=lambda x: len(x), reverse=True)

        for group in groups:
            name = group[1:]

            if auto_insert_metric_name:
                filename = filename.replace(group, name + self.CHECKPOINT_EQUALS_CHAR + "{" + name)

            # support for dots: https://stackoverflow.com/a/7934969
            filename = filename.replace(group, f"{{0[{name}]")

            if name not in metrics:
                metrics[name] = jax.Array(0)

        filename = filename.format(metrics)

        if prefix:
            filename = self.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename

    def _remove_checkpoint(self, filepath: str) -> None:
        """Calls the strategy to remove the checkpoint file."""
        os.unlink(filepath)
