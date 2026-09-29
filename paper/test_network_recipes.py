"""Fast, offline sanity checks for network_recipes.py (no solving, no pickle I/O beyond what
verify_all/reproduce_network already do -- those are exercised manually, not in CI, since they
take from seconds to ~2 minutes (verify_all) or up to 24h historically (reproduce_network)).
Run with: PYTHONPATH=.. python -m pytest test_network_recipes.py -q  (from paper/)."""
from pathlib import Path
import network_recipes as nr


def test_exactly_19_unique_recipes():
    assert len(nr.RECIPES) == 19
    assert len({r.tag for r in nr.RECIPES}) == 19
    assert len({r.net_tag for r in nr.RECIPES}) == 19


def test_covers_all_19_real_networks():
    expected = {f"crisi_dico{i}" for i in range(5)} | {f"ita_elections_dico{i}" for i in range(7)} | {f"quirinale_dico{i}" for i in range(7)}
    assert {r.net_tag for r in nr.RECIPES} == expected


def test_every_pickle_exists():
    missing = [r.pickle_name for r in nr.RECIPES if not (nr.TESTS_DIR / r.pickle_name).exists()]
    assert not missing, f"recipes reference pickles that don't exist: {missing}"


def test_confidence_is_one_of_the_documented_values():
    for r in nr.RECIPES:
        assert r.confidence in ("exact", "inferred", "partial"), r.tag


def test_resumed_recipes_name_their_source():
    for r in nr.RECIPES:
        if not r.fresh_ic:
            assert r.resume_from, f"{r.tag}: fresh_ic=False but no resume_from given"


def test_block_newton_gate_recipes_pair_with_min_rel_except_the_one_documented_exception():
    """Every recipe using block_newton_gate > 0 should also set block_newton_min_rel, EXCEPT q4,
    whose own `notes` field explicitly documents why it doesn't (it predates that kwarg)."""
    for r in nr.RECIPES:
        if r.solve_tool_kwargs.get("block_newton_gate", 0.0) > 0.0 and r.tag != "q4":
            assert "block_newton_min_rel" in r.solve_tool_kwargs, r.tag


def test_recommended_fresh_start_pairs_gate_with_min_rel():
    assert nr.RECOMMENDED_FRESH_START.get("block_newton_gate", 0.0) > 0.0
    assert "block_newton_min_rel" in nr.RECOMMENDED_FRESH_START
