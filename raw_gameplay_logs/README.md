# Raw Gameplay Logs

This folder contains raw gameplay data logs from MegaMek gameplay sessions. 
The data is stored in tab-separated value (TSV) format and needs to be processed before being used.

## Getting Started

1. Extract the contents of `DECOMPRESS IN THIS FOLDER.zip` directly into this directory.
2. The extracted files will contain raw gameplay logs in TSV format.

## Data Format

The raw data consists of tab-separated text records documenting gameplay events, including:

- Full Board description
- Planetary Conditions
- Minefields
- Unit movements
- Unit attacks
- Unit stats

Each record contains multiple fields separated by tabs, with different record types having different structures.

## Processing the Data

This raw data needs to be parsed, enriched, and transformed before it can be used effectively:

1. Parse the TSV data into structured objects using the `--parse-data` command
2. Extract meaningful features (unit positions, combat outcomes, etc.)
3. Organize data into sequential game states
4. Transform into formats suitable for analysis or ML training
