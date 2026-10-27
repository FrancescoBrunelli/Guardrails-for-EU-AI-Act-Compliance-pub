# Guardrails for EU AI Act Compliance
## Table of Contents

- [Repository Structure](#repository-structure)
- [Get Started](#get-started)
- [Versions](#versions)
- [How to Run](#how-to-run)
- [Input and Output](#input-and-output)
- [PBS Job Settings](#pbs-job-settings)
- [Cluster-specific notes](#cluster-specific-notes)

---

## Repository Structure

``` text
.
├── README.md
├── appendix         # External appendix of this study
├── plots            # Directory containing single plot files
├── results          # Directory containing both raw and processed results
├── scripts          # Directory containing cluster-specific scripts and the python script for printing LaTeX tables from the processed results
└── src              # Directory containing the benchmark scripts along with the script for analyzing the raw results
```

## Get Started
First, clone this repository to your local machine:

``` sh
git clone <repository_link>
```

(Replace `<repository_link>` with this repository's URL.)

After cloning, visit the [set up scripts](scripts/README.md#setup-scripts) section.

## Versions
Benchmarks have been run with the following versions:
- Python: `3.9.23`
- PyTorch: `2.7.1 (CUDA 11.8)`
- Transformers: `4.57.6`

## How to Run
### Run (locally):
``` sh
# Example run benchmarking LlamaGuard8B on XSTest dataset
python3 src/main.py xstest LlamaGuard8B
```

### Run (on the cluster)
To run the program on the cluster you can either:
#### Open an interactive session
``` sh
# Example using the shortGPUQ:
qsub -I -q shortGPUQ -l select=1:ncpus=28:ngpus=1:mem=256gb:walltime=07:58:00
```

And then run it in the same way you do it locally:
``` sh
# Example run benchmarking LlamaGuard8B on XSTest dataset
python3 src/main.py xstest LlamaGuard8B
```

OR
#### Run it in batch mode (using a `.pbs` script)
By creating a `.pbs` script. An example follows:
``` pbs
#!/bin/bash  
# Job name:  
#PBS -N Guardrail_benchmarking  
# Output files:  
#PBS -o runtime.out  
# Error files  
#PBS -e runtime.err  
# Queue name  
#PBS -q shortGPUQ  
# Set the maximum wall time  
#PBS -l walltime=07:58:00  
# Set the memory cpus and resources  
#PBS -l select=1:ncpus=28:ngpus=1:mem=256gb
  
# Load modules  
source envs/llm-env/bin/activate
# Select the working directory  
cd "$PBS_O_WORKDIR" # go to the directory where qsub was run
# Compile and run
python3 src/main.py <dataset_cli_name> <model_cli_name>
```

Then submit the job with:
``` sh
qsub scriptName.pbs
```
## Input and Output
### Input
The program needs to take as input the name of the dataset to benchmark on, and the name of the guardrail model to benchmark. All inputs are provided via terminal as shown in the following example:

``` sh
# Example run benchmarking GraniteGuardian2B on PatientSafetyBench dataset
python3 src/main.py psb GraniteGuardian2B
```
### Output
The output produced by the program is a `.json` file containing:
 - Input prompt
- Output response (e.g., for the ExpGuardMix dataset)
- Ground_truth (either for the input or, for the ExpGuardMix dataset both input and response)
- Category (when available)
- For each guardrail model evaluated:
	- <model_name>Output: either safe or unsafe
	- <model_name>Prob: probability score associated with the output provided by the model
	- <model_name>Time: inference time in seconds

## PBS Job Settings
By running the `.pbs` file, the script will automatically evaluate all the selected guardrail models on the specified dataset (see [datasets](src/README.md#datasets) for accepted CLI names).

### Change dataset used for benchmark
``` sh
# Example usage using XSTest dataset
qsub -v INPUT=xstest scripts/run_benchmark.pbs

# Example usage using PatientSafetyBench dataset
qsub -v INPUT=psb scripts/run_benchmark.pbs
```

### Change Memory Requested
To change the amount of RAM requested by the job when running in batch mode (via `qsub`), you need to edit the `.pbs` script and modify the line `#PBS -l select=1:ncpus=28:ngpus=1:mem=256gb` with the amount of memory you want, for example, to request 128 GB:
``` pbs
#PBS -l select=1:ncpus=28:ngpus=1:mem=128gb
```

### Change Number of CPUs Requested
To change the amount of CPUs requested by the job when running in batch mode (via qsub), you need to edit the .pbs script and modify the line `#PBS -l select=1:ncpus=28:ngpus=1:mem=256gb` with the amount of CPUs you want, for example, to request 14 CPUs:

``` pbs
#PBS -l select=1:ncpus=14:ngpus=1:mem=256gb
```

## Cluster-specific notes
**Modules to load:**
If you are using a virtual environment, activate it with:

``` sh
source envs/llm-env/bin/activate
```

**Recommended queue:** `shortGPUQ`

**Restrictions:**
- **Max concurrent jobs per user in queue:** 30
- **Max concurrent jobs per user in execution:** 30
- **Max wall time:** 7h 59min

---

> **Disclaimer:** This repository is intended for research and benchmarking purposes only. Results may vary depending on model versions, dataset versions, and evaluation settings, and should not be interpreted as absolute safety guarantees.
