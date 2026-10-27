# Scripts
## Table of Contents

- [Introduction](#introduction)
- [Setup Scripts](#setup-scripts)
- [Run Scripts](#run-scripts)
- [Python Scripts](#python-scripts)

---

## Introduction
This folder contains all the scripts needed to set up the environment, run benchmarks on different datasets, and generate LaTeX tables from analyzed results based on different performance metrics

## Setup Scripts
Scripts in this section are responsible for setting up the environment and downloading required datasets

- `download_datasets.sh` - Downloads necessary datasets from GitHub

	Usage:
``` sh
# Make sure to make it executable first:
chmod +x ./scripts/download_datasets.sh

./scripts/download_datasets.sh
```

## Run Scripts
Scripts used to run multiple tests on the compute nodes automatically

It is important to run them from the main folder of the repository

### `run_benchmark.pbs`
Benchmarks all guardrail models using the specified dataset (among the selected ones, see compressed names of [datasets](../src/README.md#datasets)), saving: Input prompt, output response (when available), ground truth, category (when available), and, for each model, output, probability score associated with the output provided by the model, and inference time expressed in seconds

**Usage:**
``` sh
# Example usage of run_benchmark.pbs
qsub -v INPUT=xstest scripts/run_benchmark.pbs
```

## Python Scripts
Organized as follows:
### `result_formatting.py`
For each file contained in the `analysis_output/` folder, generates and prints at console LaTeX tables ready to be copy-pasted. Particularly generates for each dataset the following tables:
- Single model metrics tables
- Single model, per-category metrics tables (if the dataset provides categories)
- Global metrics, for both single and combined models

**Note:** for all-unsafe or all-safe datasets, the script automatically omits metrics that are undefined (e.g. Specificity, Precision, MCC)

**Usage:**
> Requires `result_analysis.py` (in `src/`) to have been run first

``` sh
# Example usage of result_formatting.py
python3 scripts/result_formatting.py
```
