# Installation

## From PyPI

```bash
pip install pycopro
```

Optional features can be installed with extras:

```bash
pip install "pycopro[stats,plot,data]"
```

These add `statsmodels` for gene-level tests, `matplotlib` for plotting, and
`rdata` for loading example datasets.

## From source

```bash
git clone https://github.com/Zhen-Miao/copro-python.git
cd copro-python
pip install -e .
```

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24
- scipy ≥ 1.10
- pandas ≥ 2.0
- scikit-learn ≥ 1.3
- anndata ≥ 0.10

## Verify installation

```python
import copro as cp
print(cp.__version__)
```
