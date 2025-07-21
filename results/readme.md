# VersiBCB Benchmark Usage Guide

## Setup
1. Set the Python path to your working directory:
```bash
export PYTHONPATH=/path/to/your/working/folder
# Example: export PYTHONPATH=~/codes/VersiBCB_Benchmark
```

## Running Experiments
1. Generate predictions:
```bash
bash testFoundationModel.sh
```
This script runs the model to generate predictions. You can adjust model parameters in the script.

2. Evaluate results:
```bash
bash evaluate.sh
```
This script evaluates the generated predictions and produces performance metrics.

## Script Details
- `testFoundationModel.sh`: Contains model configuration and generation parameters
- `evaluate.sh`: Contains evaluation metrics and result processing logic

For detailed parameter configurations and customization options, please check the comments in each script.