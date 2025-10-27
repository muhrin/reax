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
# https://github.com/Lightning-AI/pytorch-lightning/blob/0324a20f00235c7a10a235a44326811ba42b6ae4/src/lightning/pytorch/strategies/strategy.py
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
import abc
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

import jax
import jaxtyping as jt

from .. import data as data_

if TYPE_CHECKING:
    import reax

__all__ = ("Strategy",)


BroadcastT = TypeVar("BroadcastT")
_OutT = TypeVar("_OutT")


class Strategy(abc.ABC):
    def teardown(self):
        """Shut down the strategy and free all resources."""

    @abc.abstractmethod
    def to_device(self, value: jt.PyTreeDef) -> Any:
        """Move the value to the device and return it."""

    @abc.abstractmethod
    def from_device(self, value: jt.PyTreeDef) -> Any:
        """Get a value from the device and return it."""

    @abc.abstractmethod
    def broadcast(self, obj: jt.PyTreeDef, src: int = 0) -> BroadcastT:
        """Broadcasts an object to all processes.

        :param obj: The pytree to broadcast.
        :param src: Source rank, defaults to 0.
        :type src: int, optional
        """

    @abc.abstractmethod
    def all_gather(self, obj: jt.PyTreeDef) -> Any:
        """Perform an all_gather on all processes.

        Args:
            obj: the pytree to gather.
        """

    @abc.abstractmethod
    def all_reduce(self, obj: jt.PyTree, reduce_op: str = "mean") -> jt.PyTree:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            obj: the pytree to sync and reduce
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value
        """

    @abc.abstractmethod
    def barrier(self, name: Optional[str] = None) -> None:
        """Synchronizes all processes which blocks processes until the whole group enters this
        function.

        Args:
            name: an optional name to pass into barrier.
        """

    @property
    @abc.abstractmethod
    def is_global_zero(self) -> bool:
        """Whether the current process is the rank zero process not only on the local node, but for
        all nodes."""

    @property
    @abc.abstractmethod
    def device(self) -> jax.Device:
        """Get the device used by this strategy."""

    def setup_dataloader(
        self, data: "Union[reax.DataLoader, reax.data.Dataset]"
    ) -> "reax.DataLoader":
        if isinstance(data, data_.DataLoader):
            # By default, we don't do anything to the loader
            return data

        # Presumably we are dealing with some kind of iterable that isn't a dataloader so fall back
        # to a dataloader with batch size 1 and no shuffle
        return data_.ReaxDataLoader(data)

    @abc.abstractmethod
    def compute(self, metric: "reax.Metric[_OutT]") -> _OutT:
        """Compute the value of a metric, unlike metric.compute(), in a parallel setting this method
        will compute the value across all processes."""
