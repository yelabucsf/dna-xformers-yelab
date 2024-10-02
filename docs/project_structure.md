# Project Structure

This document provides an overview of the project structure and explains the purpose of each major component.

## Overview

Project for modeling DNA sequences. It includes components for data processing, model implementation, and various analyses.

## Directory Structure

```
project_root/
├── src/
│   └── dnaformer/
├── analyses/
├── scripts/
├── data/
├── config/
├── environment/
├── docs/
├── models/
└── logs/
```

## Detailed Structure

### src/dnaformer/

This directory contains the core package code.

### analyses/

This directory contains different analysis subprojects. Each subdirectory represents a specific analysis.

- `analysis_1/`: ...
- ...

Each analysis directory typically contains:
- Python scripts for running the analysis
- R scripts for statistical tests or visualizations
- A README.md explaining the specific analysis

### scripts/

This directory contains utility scripts for the project.

- `setup_environment.sh`: Script to set up the project environment
- ...

### data/

This directory stores various data files used in the project.

- `raw/`: Original, immutable data
- `processed/`: Cleaned and processed data
- `results/`: Output from analyses

### config/

This directory contains configuration files.

### environment/

This directory contains files for environment setup and management.

- `requirements.txt`: Python package dependencies
- `renv.lock`: R package dependencies (if using renv)

### docs/

This directory contains project documentation.

- `project_structure.md`: This file

### models/

This directory stores trained model files or model checkpoints.

### logs/

This directory stores log files generated during script execution or model training.

## Setup and Usage

To set up the project environment:

1. Clone the repository
2. Run `./scripts/setup_environment.sh`
3. Activate the Python virtual environment: `source venv/bin/activate`

For more detailed setup instructions, refer to the README.md file in the project root.
