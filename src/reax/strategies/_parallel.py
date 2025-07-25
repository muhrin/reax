import abc
from typing import TYPE_CHECKING

from typing_extensions import override

from . import _strategies
from .. import data as data_

if TYPE_CHECKING:
    import reax


class ParallelStrategy(_strategies.Strategy, abc.ABC):
    @property
    @abc.abstractmethod
    def process_index(self) -> int:
        """Return the global rank of the current process."""

    @property
    @abc.abstractmethod
    def process_count(self) -> int:
        """Return the total number of ranks."""

    @override
    def setup_dataloader(self, data: "reax.DataLoader") -> "reax.DataLoader":
        if not isinstance(data, data_.DataLoader):
            data = super().setup_dataloader(data)

        seed = self.device.id
        sampler = data_.samplers.DistributedSampler(
            dataset=data,
            num_replicas=self.process_count,
            process_index=self.process_index,
            shuffle=getattr(data, "shuffle", True),
            seed=seed,
            # seed=jnp.array(0, device=self.),
            # drop_last: bool = False,
        )
        return data_.ReaxDataLoader(data, sampler=sampler)
