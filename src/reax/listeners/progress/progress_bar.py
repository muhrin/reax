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

from typing import Any, Optional

from reax import hooks

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
        """Disable the progress bar."""
        raise NotImplementedError

    def enable(self) -> None:
        """Enable the progress bar."""
        raise NotImplementedError

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print the progress bar status."""
        print(*args, **kwargs)
