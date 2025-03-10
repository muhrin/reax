# Copyright (C) 2024  Martin Uhrin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Most of this file is covered by the following license.  To find what has been modified you
# can perform a diff with the file at:
# https://github.com/Lightning-AI/pytorch-lightning/blob/0324a20f00235c7a10a235a44326811ba42b6ae4/src/lightning/pytorch/callbacks/progress/progress_bar.py
#
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Optional, Union

from lightning_utilities.core import rank_zero
from typing_extensions import override

from reax import hooks

if TYPE_CHECKING:
    import reax

__all__ = ("ProgressBar",)


class ProgressBar(hooks.TrainerListener):
    r"""The base class for progress bars in REAX. It is a :class:`~reax.listeners.TrainerListener`
    that keeps track of the batch progress in the :class:`~reax.trainer.Trainer`. You should
    implement your highly custom progress bars with this as the base class.

    Example::

        class ReaxProgressBar(ProgressBar):

            def __init__(self):
                super().__init__()  # don't forget this :)
                self.enable = True

            def disable(self):
                self.enable = False

            def on_train_batch_end(self, trainer, stage, outputs, batch, batch_idx):
                # don't forget this :)
                super().on_train_batch_end(trainer, stage, outputs, batch, batch_idx)
                percent = (batch_idx / self.total_train_batches) * 100
                sys.stdout.flush()
                sys.stdout.write(f'{percent:.01f} percent complete \r')

        bar = ReaxProgressBar()
        trainer = Trainer(callbacks=[bar])

    """

    def __init__(self) -> None:
        self._trainer: "Optional[reax.Trainer]" = None
        self._current_eval_dataloader_idx: Optional[int] = None

    @property
    def sanity_check_description(self) -> str:
        return "Sanity Checking"

    @property
    def train_description(self) -> str:
        return "Training"

    @property
    def validation_description(self) -> str:
        return "Validation"

    @property
    def test_description(self) -> str:
        return "Testing"

    @property
    def predict_description(self) -> str:
        return "Predicting"

    def disable(self) -> None:
        """You should provide a way to disable the progress bar."""
        raise NotImplementedError

    def enable(self) -> None:
        """You should provide a way to enable the progress bar.

        The :class:`~lightning.pytorch.trainer.trainer.Trainer` will call this in e.g. pre-training
        routines like the :ref:`learning rate finder
        <advanced/training_tricks:Learning Rate Finder>`. to temporarily enable and disable the
        training progress bar.

        """
        raise NotImplementedError

    def print(self, *args: Any, **kwargs: Any) -> None:
        """You should provide a way to print without breaking the progress bar."""
        print(*args, **kwargs)

    @override
    def setup(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        self._trainer = trainer
        if not trainer.is_global_zero:
            self.disable()

    def get_metrics(
        self, trainer: "reax.Trainer", *_
    ) -> dict[str, Union[int, str, float, dict[str, float]]]:
        r"""Combines progress bar metrics collected from the trainer with standard metrics from
        get_standard_metrics. Implement this to override the items displayed in the progress bar.

        Here is an example of how to override the defaults:

        .. code-block:: python

            def get_metrics(self, trainer, model):
                # don't show the version number
                items = super().get_metrics(trainer, model)
                items.pop("v_num", None)
                return items

        Return:
            Dictionary with the items to be displayed in the progress bar.

        """
        standard_metrics = get_standard_metrics(trainer)
        pbar_metrics = trainer.progress_bar_metrics
        duplicates = list(standard_metrics.keys() & pbar_metrics.keys())
        if duplicates:
            rank_zero.rank_zero_warn(
                f"The progress bar already tracks a metric with the name(s) "
                f"'{', '.join(duplicates)}' and `self.log('{duplicates[0]}', ..., prog_bar=True)` "
                f"will overwrite this value. If this is undesired, change the name or override "
                f"`get_metrics()` in the progress bar callback."
            )

        return {**standard_metrics, **pbar_metrics}


def get_standard_metrics(trainer: "reax.Trainer") -> dict[str, Union[int, str]]:
    r"""Returns the standard metrics displayed in the progress bar. Currently, it only includes the
    version of the experiment when using a logger.

    .. code-block::

        Epoch 1:   4%|▎         | 40/1095 [00:03<01:37, 10.84it/s, v_num=10]

    Return:
        Dictionary with the standard metrics to be displayed in the progress bar.

    """
    items_dict: dict[str, Union[int, str]] = {}
    if trainer.loggers:

        if (version := _version(trainer.loggers)) not in ("", None):
            if isinstance(version, str):
                # show last 4 places of long version strings
                version = version[-4:]
            items_dict["v_num"] = version

    return items_dict


def _version(loggers: list[Any], separator: str = "_") -> Union[int, str]:
    if len(loggers) == 1:
        return loggers[0].version
    # Concatenate versions together, removing duplicates and preserving order
    return separator.join(dict.fromkeys(str(logger.version) for logger in loggers))
