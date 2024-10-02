# DNA Language Modeling

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development](#development)
- [AWS S3 Sync](#aws-s3-sync)
- [Contributing](#contributing)
- [License](#license)

## Installation

This project uses Python 3.8+ and R 4.0+. To set up the project environment:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Run the setup script:
   ```bash
   ./scripts/setup_environment.sh
   ```

## Usage

1. Activate the Python virtual environment:
   ```
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Run an analysis:
   ```
   python analyses/example_usage.py
   ```

3. To deactivate the virtual environment when you're done:
   ```
   deactivate
   ```

## Project Structure

For a detailed explanation of the project structure, see [project_structure.md](docs/project_structure.md)
in `docs/`.

TLDR:
- Custom python modules that include models like transformers with rotary embeddings are in `./src/code`.
- Scripts to analyze the data and generate the results are in `./analyses`.
- Trained models can be stored in `./models`.
- Custom bash and R reusable scripts can be stored `./scripts`.

## Development

When adding new code to the project, please follow the guidelines in [adding_new_code_guide.md](docs/adding_new_code_guide.md). This guide provides step-by-step instructions on how to add new files, classes, or functions to the `src/code` directory and ensure they are properly integrated into the project structure.

## AWS S3 Sync

This project includes functionality to sync data and models with an AWS S3 bucket.

### Prerequisites

1. Install the AWS CLI and configure it with your credentials.
2. Ensure you have the virtual environment `venv` available after running
   `./scripts/setup_environment.sh` in [Installation](#installation)

### Usage

To sync data and models with the S3 bucket `dna-xformers-yelab`:

```bash
./scripts/run_s3_sync.sh <direction> <main_dir> [subdirectory]
```

Where:
- `<direction>` is either `to_s3` or `from_s3`
- `<main_dir>` is either `data` or `models`
- `[subdirectory]` is an optional specific subdirectory you want to sync

Examples:
- To sync the entire `data` directory to S3:
  ```
  ./scripts/run_s3_sync.sh to_s3 data
  ```
- To sync a specific subdirectory `data/results/analysis` from S3:
  ```
  ./scripts/run_s3_sync.sh from_s3 data results/analysis
  ```

The scripts will use the S3 bucket named `dna-xformers-yelab` for all sync operations.

## Note on PyTorch

This project requires PyTorch, but it's not installed via pip. It's assumed that you have a system-wide
installation of PyTorch optimized for your hardware. If you don't have PyTorch installed, please install
it according to the official PyTorch installation guide: https://pytorch.org/get-started/locally/
