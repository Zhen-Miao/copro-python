"""AnnData constructor and modern input-validation tests."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

anndata = pytest.importorskip("anndata")

from copro.core import CoProMulti, CoProSingle, create_copro, from_anndata


def test_from_anndata_preserves_sparse_expression_and_alignment():
    cells = [f"cell_{i}" for i in range(12)]
    genes = [f"gene_{j}" for j in range(5)]
    X = sparse.csr_matrix(np.arange(60, dtype=float).reshape(12, 5))
    obs = pd.DataFrame(
        {
            "type": ["A"] * 6 + ["B"] * 6,
            "sample": ["s1"] * 4 + ["s2"] * 8,
        },
        index=cells,
    )
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    adata.obsm["spatial"] = np.column_stack((np.arange(12), np.arange(12) ** 2))

    obj = from_anndata(
        adata, cell_type_key="type", slide_key="sample", max_cell=100
    )
    assert isinstance(obj, CoProMulti)
    assert sparse.issparse(obj.normalized_data)
    assert obj.gene_names == genes
    assert obj.meta_data.index.tolist() == cells
    np.testing.assert_array_equal(obj.cell_types, obs["type"].to_numpy())
    np.testing.assert_array_equal(
        obj.meta_data["slideID"].to_numpy(), obs["sample"].to_numpy()
    )


def test_from_anndata_single_and_casefolded_coordinates():
    adata = anndata.AnnData(
        X=np.ones((6, 3)),
        obs=pd.DataFrame({"type": ["A"] * 6}),
    )
    adata.obsm["spatial"] = pd.DataFrame(
        {"X": np.arange(6), "Y": np.arange(6)}, index=adata.obs_names
    )
    with pytest.warns(UserWarning, match="Standardizing"):
        obj = from_anndata(adata, cell_type_key="type")
    assert isinstance(obj, CoProSingle)
    assert {"x", "y"}.issubset(obj.location_data.columns)


def test_create_copro_rejects_nonfinite_expression_and_slide_mismatch():
    X = np.ones((8, 3))
    loc = pd.DataFrame({"x": np.arange(8), "y": np.arange(8)})
    meta = pd.DataFrame({"slideID": ["existing"] * 8})
    types = np.array(["A"] * 8)
    X[2, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        create_copro(X, loc, meta, types)

    X[2, 1] = 0
    with pytest.raises(ValueError, match="do not match"):
        create_copro(X, loc, meta, types, slide_id=np.array(["other"] * 8))
