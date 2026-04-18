"""Download example datasets for CoPro tutorials."""

from __future__ import annotations

import os
import urllib.request


def copro_download_data(
    dataset: str,
    destdir: str | None = None,
    tag: str = "data-v1",
    overwrite: bool = False,
) -> dict:
    """Download and load an example dataset from GitHub Releases.

    Downloads pre-processed, subsampled datasets attached to the CoPro
    R package GitHub Releases and converts them to Python objects.

    Parameters
    ----------
    dataset : str
        One of ``"colon_d3"``, ``"colon_d9"``, ``"kidney"``,
        ``"organoid"``, or ``"brain_merfish"``.
    destdir : str or None
        Directory to cache the downloaded file.  Defaults to
        ``~/.copro/data``.
    tag : str
        GitHub release tag (default ``"data-v1"``).
    overwrite : bool
        Re-download even if the file already exists locally.

    Returns
    -------
    dict
        Dataset contents as a dictionary.  Keys match the R list names
        (e.g. ``"normalizedData"``, ``"locationData"``, ``"metaData"``,
        ``"cellTypes"``).  Matrices are returned as numpy arrays or
        xarray DataArrays, data frames as pandas DataFrames, and
        vectors as numpy arrays.

    Notes
    -----
    Requires the ``rdata`` package (``pip install rdata``).
    """
    try:
        import rdata
    except ImportError:
        raise ImportError(
            "rdata is required to load example data. "
            "Install with: pip install rdata"
        )

    valid = {"colon_d3", "colon_d9", "kidney", "organoid", "brain_merfish"}
    if dataset not in valid:
        raise ValueError(f"dataset must be one of {valid}, got '{dataset}'")

    file_name = f"copro_{dataset}.rds"

    if destdir is None:
        destdir = os.path.join(os.path.expanduser("~"), ".copro", "data")
    os.makedirs(destdir, exist_ok=True)

    dest_file = os.path.join(destdir, file_name)

    if os.path.exists(dest_file) and not overwrite:
        print(f"Using cached file: {dest_file}")
    else:
        url = (
            f"https://github.com/Zhen-Miao/CoPro/releases/download/"
            f"{tag}/{file_name}"
        )
        print(f"Downloading {file_name} from GitHub Release '{tag}'...")
        urllib.request.urlretrieve(url, dest_file)
        print(f"Downloaded to: {dest_file}")

    result = rdata.read_rds(dest_file)

    import numpy as np
    import pandas as pd

    out = {}
    if isinstance(result, dict):
        for key, val in result.items():
            key = str(key)  # np.str_ → str
            try:
                import xarray as xr
                if isinstance(val, xr.DataArray):
                    if val.ndim == 2:
                        # Matrix with row/column names → DataFrame
                        row_names = val.coords[val.dims[0]].values
                        col_names = val.coords[val.dims[1]].values
                        out[key] = pd.DataFrame(
                            val.values, index=row_names, columns=col_names,
                        )
                    elif val.ndim == 1:
                        out[key] = val.values
                    else:
                        out[key] = val.values
                    continue
            except ImportError:
                pass
            out[key] = val
    else:
        out["data"] = result

    return out
