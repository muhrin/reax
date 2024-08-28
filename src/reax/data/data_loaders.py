from typing import Optional

from . import _types, collate, fetchers, samplers


class GenericDataLoader(_types.DataLoader):
    def __init__(
        self,
        dataset: _types.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn: Optional[_types.CollateFn] = None,
    ):
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
        return self._batch_size

    def __iter__(self):
        for indices in self._sampler:
            yield self._fetcher.fetch(indices)
