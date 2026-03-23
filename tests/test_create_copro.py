"""Tests for create_copro auto-K blocking constructor."""

import numpy as np
import pandas as pd
import pytest

from copro.core import CoProSingle, CoProMulti, create_copro


def _make_data(n_cells, n_genes=20, slide_id=None):
    """Helper: generate random test data."""
    rng = np.random.RandomState(42)
    normalized_data = rng.randn(n_cells, n_genes)
    location_data = pd.DataFrame({
        "x": rng.rand(n_cells),
        "y": rng.rand(n_cells),
    })
    meta_data = pd.DataFrame({"cell_id": [f"c{i}" for i in range(n_cells)]})
    cell_types = np.array(["TypeA", "TypeB"] * (n_cells // 2))
    return normalized_data, location_data, meta_data, cell_types


class TestCreateCoproSingleSmall:
    """Single slide, n_cells <= max_cell → CoProSingle."""

    def test_returns_single(self):
        data, loc, meta, ct = _make_data(100)
        obj = create_copro(data, loc, meta, ct, max_cell=200)
        assert isinstance(obj, CoProSingle)

    def test_no_slide_id_column(self):
        data, loc, meta, ct = _make_data(100)
        obj = create_copro(data, loc, meta, ct, max_cell=200)
        assert "slideID" not in obj.meta_data.columns


class TestCreateCoproSingleOversized:
    """Single slide, n_cells > max_cell → CoProMulti with _blk suffixes."""

    def test_returns_multi(self):
        data, loc, meta, ct = _make_data(200)
        obj = create_copro(data, loc, meta, ct, max_cell=80)
        assert isinstance(obj, CoProMulti)

    def test_slide_ids_have_blk_suffix(self):
        data, loc, meta, ct = _make_data(200)
        obj = create_copro(data, loc, meta, ct, max_cell=80)
        slides = obj.meta_data["slideID"].unique()
        assert all("_blk" in s for s in slides)

    def test_all_blocks_within_limit(self):
        data, loc, meta, ct = _make_data(500)
        obj = create_copro(data, loc, meta, ct, max_cell=120)
        counts = obj.meta_data["slideID"].value_counts()
        assert counts.max() <= 120


class TestCreateCoproMultiNoOversized:
    """Multiple slides, all <= max_cell → CoProMulti with original IDs."""

    def test_preserves_original_ids(self):
        data, loc, meta, ct = _make_data(100)
        slide_id = np.array(["s1"] * 50 + ["s2"] * 50)
        obj = create_copro(data, loc, meta, ct, slide_id=slide_id, max_cell=200)
        assert isinstance(obj, CoProMulti)
        assert set(obj.meta_data["slideID"].unique()) == {"s1", "s2"}


class TestCreateCoproMultiSomeOversized:
    """Multiple slides, some > max_cell → refined IDs."""

    def test_only_oversized_slide_gets_split(self):
        data, loc, meta, ct = _make_data(200)
        slide_id = np.array(["small"] * 30 + ["big"] * 170)
        obj = create_copro(data, loc, meta, ct, slide_id=slide_id, max_cell=80)
        slides = obj.meta_data["slideID"].unique()
        # "small" stays, "big" gets split
        assert "small" in slides
        assert any("big_blk" in s for s in slides)
        # all blocks within limit
        counts = obj.meta_data["slideID"].value_counts()
        assert counts.max() <= 80


class TestCreateCoproValidation:
    """Input validation."""

    def test_dimension_mismatch_raises(self):
        data, loc, meta, ct = _make_data(100)
        with pytest.raises(ValueError, match="dimensions"):
            create_copro(data, loc, meta, ct[:50])

    def test_slide_id_length_mismatch_raises(self):
        data, loc, meta, ct = _make_data(100)
        with pytest.raises(ValueError, match="dimensions"):
            create_copro(data, loc, meta, ct, slide_id=np.array(["s1"] * 50))


class TestCreateCoproReproducibility:
    """Same seed → same partition."""

    def test_same_seed_same_result(self):
        data, loc, meta, ct = _make_data(300)
        obj1 = create_copro(data, loc, meta.copy(), ct, max_cell=80, seed=42)
        obj2 = create_copro(data, loc, meta.copy(), ct, max_cell=80, seed=42)
        np.testing.assert_array_equal(
            obj1.meta_data["slideID"].values,
            obj2.meta_data["slideID"].values,
        )

    def test_different_seed_different_result(self):
        data, loc, meta, ct = _make_data(300)
        obj1 = create_copro(data, loc, meta.copy(), ct, max_cell=80, seed=1)
        obj2 = create_copro(data, loc, meta.copy(), ct, max_cell=80, seed=99)
        # Not guaranteed to differ, but overwhelmingly likely with 300 cells
        assert not np.array_equal(
            obj1.meta_data["slideID"].values,
            obj2.meta_data["slideID"].values,
        )
