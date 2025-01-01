from typing import Optional

from . import _types, collate, fetchers, samplers

__all__ = ("GenericDataLoader",)


class GenericDataLoader(_types.DataLoader):
    def __init__(
        self,
        dataset: _types.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn: Optional[_types.CollateFn] = None,
    ):
        """Init function."""
        self._batch_size = batch_size
        self._dataset = dataset
        self._sampler = samplers.create_sampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        if collate_fn is None:
            collate_fn = collate.get_default_collator().collate

        self._fetcher = fetchers.create_fetcher(dataset, collate_fn=collate_fn)

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self._batch_size

    def __iter__(self):
        """Iter function."""
        for indices in self._sampler:
            yield self._fetcher.fetch(indices)
