# MM-Caspar-Trainer
## CASPAR Neural Network Training Framework

This repository contains a framework for training the CASPAR neural network model using MLFlow for experiment tracking.
The setup uses Docker Compose to provide a complete MLFlow stack with MinIO for artifact storage and PostgreSQL for the 
backend database.

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.11 or later
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the MLFlow infrastructure:
   ```bash
   docker-compose up -d
   ```

   This will start:
   - MinIO (S3-compatible storage) at http://localhost:9000 (UI at http://localhost:9001)
   - PostgreSQL database at localhost:5432
   - MLFlow server at http://localhost:9002

   Now you need to setup the MinIO bucket for MLFlow artifacts:
   - Access the MinIO web console at http://localhost:9001
   - Login with the following credentials:
     - Access Key: mlops-demo
     - Secret Key: mlops-demo
   - Select the option "Buckets" under the `Administrator` heading and there you create a bucket named `mlops`
   - Open it and inside you change its access policy to `Public` so that the MLFlow server can access it.
   - Now you can access the MLFlow UI at http://localhost:9002 and your experiments are going to correctly be logged.

## Data Preparation

Before training the model, you need to prepare the datasets. The `--data` flag initializes a new batch of data files by:
1. Loading raw datasets from the directory specified in `caspar.config.DATASETS_DIR`
2. Extracting features using the `FeatureExtractor` class
3. Splitting the data into training, validation, and test sets
4. Saving the processed datasets for future use by training processes

To prepare the data:

```bash
python __main__.py --data
```

## Model Training

### Basic Training

To train the model with default parameters:

```bash
python __main__.py
```

### Customizing Training Parameters

You can customize various training parameters:

```bash
python __main__.py --epochs 100 --batch-size 64 --learning-rate 0.01 --dropout-rate 0.3 --hidden-layers 354 354
```

### Hyperparameter Optimization

The framework includes hyperparameter optimization using Optuna:

```bash
python __main__.py --optimize --n-trials 100 --n-jobs -1
```

Options:
- `--optimize`: Enable hyperparameter optimization
- `--n-trials`: Number of trials for optimization (default: 100)
- `--n-jobs`: Number of parallel jobs (-1 uses all cores)

### Experiment Tracking

Experiments are tracked using MLFlow. You can customize the experiment name:

```bash
python __main__.py --experiment-name "my-experiment" --run-name "test-run-1"
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mekfile` | str | From config | Path to Mek file (txt) |
| `--data` | flag | - | Recompile the datasets |
| `--epochs` | int | From config | Number of training epochs |
| `--dropout-rate` | float | From config | Dropout rate |
| `--hidden-layers` | list of int | From config | Hidden layers configuration |
| `--batch-size` | int | From config | Batch size for training |
| `--experiment-name` | str | From config | MLflow experiment name |
| `--optimize` | flag | - | Run hyperparameter optimization |
| `--n-trials` | int | 100 | Number of trials for hyperparameter optimization |
| `--test-size` | float | From config | Percent of dataset to use for testing |
| `--n-jobs` | int | -1 | Number of parallel jobs for optimization |
| `--learning-rate` | float | From config | Learning rate for optimization |
| `--run-name` | str | None | Name of the run |

## Workflow Example

A typical workflow might look like:

1. Start the MLFlow infrastructure:
   ```bash
   docker-compose up -d
   ```

2. Prepare the datasets:
   ```bash
   python __main__.py --data
   ```

3. [Optional] Run hyperparameter optimization:
   ```bash
   python __main__.py --optimize --n-trials 50 --experiment-name "hyperparameter-search"
   ```

4. Train the model with the parameters:
   ```bash
   python __main__.py --hidden-layers 354 354 --dropout-rate 0.06 --learning-rate 0.01 --epochs 50 --experiment-name final-model --run-name model-v1
   ```

5. View the results in the MLFlow UI:
   ```
   http://localhost:9002
   ```

6. Convert the model to TFLite format to deploy on MegaMek:
   ```bash
   python __main__.py --convert-model model-v1
   ```
   
## Architecture

The framework consists of several components:

- **Data Loading**: Loads raw datasets from the configured directory
- **Feature Extraction**: Extracts relevant features from the raw data
- **Model Definition**: Defines the neural network architecture
- **Training**: Handles model training and evaluation
- **Hyperparameter Optimization**: Optimizes model architecture and training parameters
- **MLFlow Integration**: Tracks experiments, parameters, metrics, and artifacts

## MLFlow Integration

The framework uses MLFlow for experiment tracking. All training runs are logged to the MLFlow server, including:

- Parameters: Model architecture, training parameters, etc.
- Metrics: Loss, accuracy, etc.
- Artifacts: Model files, plots, etc.

You can access the MLFlow UI at http://localhost:9002 to view and compare experiments.

## MinIO Integration

MinIO provides S3-compatible storage for MLFlow artifacts. You can access the MinIO web console at http://localhost:9001 using:
- Username: mlops-demo
- Password: mlops-demo

## Notes

- The framework automatically sets up MLFlow with the tracking URI from the configuration.
- Model checkpoints and artifacts are stored in MinIO and can be accessed via the MLFlow UI.
- Use the `--run-name` parameter to give meaningful names to your experiments.

## Customizing the Model Behavior

If you want to significantly change how the bot works, you can modify or extend the `FeatureExtractor` class. 
This class is responsible for transforming raw game data into features that the neural network can learn from.

The feature extractor:
- Processes raw unit actions and game states
- Calculates meaningful features like unit health, position, threat levels, etc.
- Determines how different game aspects are represented to the model

To customize the model behavior:

1. Locate the `feature_extractor.py` file in the project
2. Modify existing feature calculations or add new features
3. Ensure you update the feature list in the `__init__` method if adding new features
4. Re-run the data preparation step with `python __main__.py --data` to regenerate datasets with your modified features

Important methods you might want to modify:
- `extract_features()`: The main method that processes all features
- Various calculation methods (e.g., `_calculate_unit_role()`, `_calculate_threat_by_role()`)
- `reward_calculator()`: Determines how actions are rewarded during training

Remember that any changes to the feature extractor must be reflected in the game client in the Caspar feature extractor 
to ensure consistency between training and gameplay, otherwise the model will not perform as expected.
