
# MM-Caspar-Trainer
## CASPAR Neural Network Training Framework

This repository contains a framework for training the CASPAR neural network model using MLFlow for experiment tracking.
The setup uses Docker Compose to provide a complete MLFlow stack with MinIO for artifact storage and PostgreSQL for the 
backend database.

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.11.9
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
1. Loading raw gameplay logs from the directory specified in `caspar.config.RAW_GAMEPLAY_LOGS_DIR`
2. Extracting features using the `FeatureExtractor` class
3. Splitting the data into training, validation, and test sets
4. Saving the processed datasets for future use by training processes

To prepare the data, it will make the tagged datasets available in the `datasets_tagged` directory. Then we prepare the training data, it will generate the `training_data.csv` file and numpy array bins with the data
for training, validation and test sets:

```bash
# Generate tagged datasets
python -m caspar --parse-datasets

# Generate training data (using undersampling by default)
python -m caspar --extract-features

# Generate training data with oversampling
python -m caspar --extract-features --oversample
```

### Data Directory Structure
The data directory should contain the following structure:

```
caspar/
├── data (generated, used for training models)
│   ├── _xx_xxxx_feature_statistics.csv 
│   ├── LICENSE
│   ├── README.md
│   ├── class_info.json
│   ├── min_max_feature_normalization.csv
│   ├── x_test.npy
│   ├── x_train.npy
│   ├── x_val.npy
│   ├── y_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── training_data.csv
│   └── ...
├── raw_gameplay_logs
│   ├── LICENSE
│   ├── README.md
│   ├── human_games
│   │   ├── dataset q=100 a=0043 p=02 u=006 bv=009911 d=2025-04-15 t=15-18-20_7829.tsv
│   │   └── ...
│   └── human_vs_princess_games
│       ├── dataset q=007 a=0325 p=03 u=027 bv=042689 d=2025-03-26 t=02-20-57_9364.tsv
│       └── ...
├── resources (used to enrich older datasets, generated from MegaMek)
│   ├── LICENSE
│   └── meks.tsv
└── datasets_tagged (generated)
    ├── LICENSE
    ├── README.md
    ├── tagged dataset q=006 a=0033 p=01 u=004 bv=000900 d=2025-03-16 t=17-44-18_9083 id=252.json
    ├── tagged dataset q=100 a=0144 p=02 u=012 bv=019971 d=2025-04-04 t=19-37-31_1047 id=148.json
    └── ...
```


## Model Training

You can manually set the configuration of the model by passing the parameters in the command line or by editing the `config.py` file.

```python
# config.py

# Model settings
MODEL_CONFIG = {
    "hidden_layers": [1036, 1036, 130, 518, 340, 130, 65, 130],
    "dropout_rate": 0.18344289381176082,
    "learning_rate": 0.016544355731452045
}

# Training settings
TRAINING_CONFIG = {
    "test_size": 0.1,
    "validation_size": 0.1,
    "epochs": 500,
    "batch_size": 128,
}

```

Then we train the model with the data we just generated.

```bash
# Train with default configuration
python -m caspar 

# Train with custom parameters
python -m caspar --epochs 100 --batch-size 64 --learning-rate 0.01 --dropout-rate 0.3 --hidden-layers 354 354

# Train with custom experiment name
python -m caspar --experiment-name "my-experiment" --run-name "test-run-1"
```

### Customizing Training Parameters

You can customize various training parameters:

```bash
python -m caspar --epochs 100 --batch-size 64 --learning-rate 0.01 --dropout-rate 0.3 --hidden-layers 354 354
```

### Hyperparameter Optimization

The framework includes hyperparameter optimization using Optuna:

```bash
python -m caspar --optimize --n-trials 10 --n-jobs -1
```

Options:
- `--optimize`: Enable hyperparameter optimization
- `--n-trials`: Number of trials for optimization (default: 100)
- `--n-jobs`: Number of parallel jobs (-1 uses all cores)

### Experiment Tracking

Experiments are tracked using MLFlow. You can customize the experiment name:

```bash
python -m caspar --experiment-name "my-experiment" --run-name "test-run-1"
```

## Command Line Arguments

The command line arguments are organized into mutually exclusive groups:

### Dataset Handling
| Argument | Type | Description |
|----------|------|-------------|
| `--name-datasets` | flag | Rename the datasets in the datasets directory |
| `--parse-datasets` | flag | Compile the datasets with pre-tags from the raw game action data |
| `--extract-features` | flag | Extract features from datasets and create untagged training data |
| `--test-size` | float | Percent from 0 to 1 of the dataset to use for testing |
| `--validation-size` | float | Percent from 0 to 1 of the dataset to use for validation |
| `--oversample` | flag | Oversample the training data to balance the classes (default behavior is to undersample) |

### Test Model
| Argument | Type | Description |
|----------|------|-------------|
| `--s3-model` | str | Path to a trained model stored in S3 to load and test |

### Experiment Configuration
| Argument | Type | Description |
|----------|------|-------------|
| `--experiment-name` | str | MLflow experiment name |
| `--run-name` | str | Name of the run |
| `--feature-correlation` | flag | Check feature correlation |
| `--model-name` | str | Name of the model |

### Training And Model Architecture
| Argument | Type | Description |
|----------|------|-------------|
| `--epochs` | int | Number of training epochs |
| `--dropout-rate` | float | Dropout rate |
| `--batch-size` | int | Batch size for training |
| `--learning-rate` | float | Learning rate for optimization |
| `--hidden-layers` | list of int | Hidden layers configuration |

### Hyperparameter Search
| Argument | Type | Description |
|----------|------|-------------|
| `--optimize` | flag | Run hyperparameter optimization |
| `--n-trials` | int | Number of trials for hyperparameter optimization |
| `--n-jobs` | int | Number of parallel jobs for optimization (-1 uses all cores) |

### Other Arguments
| Argument | Type | Description |
|----------|------|-------------|
| `--mekfile` | str | Path to Mek file (txt) |

**Note**: Arguments within each group are mutually exclusive from arguments in other groups. For example, you cannot use `--extract-features` together with `--optimize`.
The workflow example section should be updated to match the current command-line arguments and workflow. Here's an updated version:

## Workflow Example

A typical workflow might look like:

1. Start the MLFlow infrastructure and set up the MinIO bucket:
   ```bash
   docker-compose up -d
   ```
  Set up the MinIO bucket for MLFlow artifacts:
   - Access the MinIO web console at http://localhost:9001
   - Login with the following credentials:
     - Access Key: mlops-demo
     - Secret Key: mlops-demo
   - Select the option "Buckets" under the `Administrator` heading and there you create a bucket named `mlops`
   - Open it and inside you change its access policy to `Public` so that the MLFlow server can access it.
   - Now you can access the MLFlow UI at http://localhost:9002 and your experiments are going to correctly be logged.

2. Prepare the datasets:
   ```bash
   # Generate tagged datasets
   python -m caspar --parse-datasets

   # Generate training data (using undersampling by default)
   python -m caspar --extract-features

   # Or use oversampling to balance classes
   python -m caspar --extract-features --oversample
   ```

3. [Optional] Run hyperparameter optimization:
   ```bash
   python -m caspar --optimize --n-trials 50 --n-jobs -1 --experiment-name "hyperparameter-search"
   ```

4. Train the model with the parameters:
   ```bash
   python -m caspar --hidden-layers 1036 1036 130 518 340 130 65 130 --dropout-rate 0.18 --learning-rate 0.016 --epochs 500 --experiment-name final-model --run-name model-v1
   ```

5. View the results in the MLFlow UI:
   ```
   http://localhost:9002
   ```

6. Test a model stored in S3 (actually it is stored locally in minIO, but anyway): 
   ```bash
   python -m caspar --s3-model s3://mybucket/path/to/model.h5
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
