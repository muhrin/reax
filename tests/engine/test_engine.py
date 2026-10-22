import reax
from reax.lightning import rank_zero


def test_engine_sets_rank_zero_rank(monkeypatch):
    """The engine sets the rank used by the rank-zero utilities from its strategy."""
    # Pretend a previous run left a different rank behind
    monkeypatch.setattr(rank_zero.rank_zero_only, "rank", 3, raising=False)

    engine = reax.Engine(logger=False)

    assert engine.process_index == 0
    assert rank_zero.get_rank() == 0
