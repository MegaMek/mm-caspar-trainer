# Copyright (C) 2025-2025 The MegaMek Team. All Rights Reserved.
#
# This file is part of MM-Caspar-Trainer.
#
# MM-Caspar-Trainer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License (GPL),
# version 3 or (at your option) any later version,
# as published by the Free Software Foundation.
#
# MM-Caspar-Trainer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# A copy of the GPL should have been included with this project;
# if not, see <https://www.gnu.org/licenses/>.
#
# NOTICE: The MegaMek organization is a non-profit group of volunteers
# creating free software for the BattleTech community.
#
# MechWarrior, BattleMech, `Mech and AeroTech are registered trademarks
# of The Topps Company, Inc. All Rights Reserved.
#
# Catalyst Game Labs and the Catalyst Game Labs logo are trademarks of
# InMediaRes Productions, LLC.
import json
import os
import tempfile

import boto3
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from tqdm import tqdm

from caspar.config import RAW_GAMEPLAY_LOGS_DIR, MEK_FILE, TRAINING_CONFIG, DOTENV_PATH, DATA_DIR, \
    DATASETS_TAGGED_DIR
from caspar.data.data_loader import load_datasets, load_tagged_datasets_classifier, \
    DataLoader, load_dataset_from_file
from caspar.data.feature_extractor import ClassifierFeatureExtractor
from caspar.data.tagger import tag_action
from caspar.data.training_dataset_processor import ClassificationTrainingDatasetProcessor

load_dotenv(DOTENV_PATH)


def load_model_from_s3(address):
    print(f"Loading model from {address}...")
    # Parse S3 path
    s3_path = address.replace("s3://", "")
    bucket_name = s3_path.split("/")[0]
    key = "/".join(s3_path.split("/")[1:])

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name
        # Download model to the temporary file
        s3_client.download_file(Bucket=bucket_name, Key=key, Filename=temp_path)
        # Load model from the temporary file
        return tf.keras.models.load_model(temp_path)


def count_players(unit_states):
    players = set()
    for unit_state in unit_states:
        if unit_state.get('player_id') is not None:
            players.add(unit_state['player_id'])
    return len(players)


def count_units(unit_states):
    units = set()
    for unit_state in unit_states:
        if unit_state.get('entity_id'):
            units.add(unit_state['entity_id'])
    return len(units)


def count_bv(unit_states):
    bv = set()
    for unit_state in unit_states:
        if unit_state.get('bv'):
            bv.add(unit_state['bv'])
    return sum(bv)


def count_bot_players(unit_actions):
    bot_players = set(
        action['player_id'] for action in unit_actions
        if action.get('is_bot') == 1
    )

    return len(bot_players)

def calculate_quality(unit_actions, game_states, game_board):
    """
    Quality is calculated as the ratio of human players to bot players
    in the game. The higher the ratio, the better the quality.
    """

    human_players = set(
        action['player_id'] for action in unit_actions
        if action.get('is_bot') == 0
    )

    unique_units = {}
    for game_states_array in game_states:
        for game_state in game_states_array:
            unit_key = str(game_state.get('entity_id'))
            if unit_key not in unique_units:
                unique_units[unit_key] = {
                    'isBot': game_state.get('player_id') not in human_players,
                    'bv': game_state.get('bv', 0)
                }

    human_bv = 0
    bot_bv = 0

    for unit in unique_units.values():
        if unit['isBot']:
            bot_bv += unit['bv']
        else:
            human_bv += unit['bv']

    # Mark datasets with low human player presence as lower quality
    adjusted_bot_bv = bot_bv * 4
    if adjusted_bot_bv == 0:
        return 100

    return min((human_bv / adjusted_bot_bv * 100), 99)


def make_tagged_datasets():
    unit_actions_tuple, game_states_tuple, game_boards_tuple, file_names = load_datasets(double_blind=False)
    tag_and_persist_dataset(game_boards_tuple, game_states_tuple, unit_actions_tuple, file_names)

    offset = len(unit_actions_tuple)
    unit_actions_tuple, game_states_tuple, game_boards_tuple, file_names = load_datasets(double_blind=True)
    tag_and_persist_dataset(game_boards_tuple, game_states_tuple, unit_actions_tuple, file_names, offset=offset)


def tag_and_persist_dataset(game_boards_tuple, game_states_tuple, unit_actions_tuple, file_names, offset = 0):
    with tqdm(total=len(unit_actions_tuple), desc="Tagging datasets") as t:
        for i in range(len(unit_actions_tuple)):
            file_path = file_names[i]

            unit_states = [unit_state for unit_state in game_states_tuple[i][1][0]]
            players = count_players(unit_states)
            bots = count_bot_players(unit_states)
            units = count_units(unit_states)
            bv = count_bv(unit_states)
            quality = int(calculate_quality(unit_actions_tuple[i][1], game_states_tuple[i][1], game_boards_tuple[i][1]))
            
            name = create_file_name(bv, file_path, players, quality, unit_actions_tuple[i][1], units,
                                    prefix="tagged dataset", extension="json", postfix=f"id={i + offset}")

            file = os.path.join(
                DATASETS_TAGGED_DIR,
                name
            )

            filepath = os.path.join(DATASETS_TAGGED_DIR, file)
            desc_text = filepath[-60:] + " " * (60 - len(filepath[-60:]))

            t.set_description("Tagging: " + desc_text)
            t.update()

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "unitActions": unit_actions_tuple[i][1],
                    "gameStates": game_states_tuple[i][1],
                    "gameBoard": game_boards_tuple[i][1],
                    "tags": tag_action(unit_actions_tuple[i][1], game_states_tuple[i][1], game_boards_tuple[i][1]),
                    "notes": "original file name: " + file_path,
                    "properties": {
                        "id": i+offset,
                        "quality": quality,
                        "players": players,
                        "bots": bots,
                        "human_players": players - bots,
                        "board_size": game_boards_tuple[i][1],
                        "units": units,
                        "bv": bv,
                    },
                }, f)


def create_file_name(bv, file_path, players, quality, unit_actions, units, prefix, extension, postfix: str = ""):
    date_time = read_first_line(file_path).split(" ")[-1].strip().replace(":", "-")
    _date, _time = date_time.split("T")
    millis = _time.split(".")[1]
    _time = _time.split(".")[0] + "_" + millis[:4]
    if postfix:
        postfix = " " + postfix
    return f"{prefix} q={quality:03d} a={len(unit_actions):04d} p={players:02d} u={units:03d} bv={bv:06d} d={_date} t={_time}{postfix}.{extension}"


def make_test_train_val_data_classifier(oversample: bool):
    unit_actions, game_states, game_boards, tags = load_tagged_datasets_classifier()
    feature_extractor = ClassifierFeatureExtractor()

    x_acc = []
    y_acc = []

    for i in range(len(unit_actions)):
        x, y = feature_extractor.extract_classification_features(unit_actions[i][1], game_states[i][1], game_boards[i][1], tags[i][1])
        x_acc.append(x)
        y_acc.append(y)

    # Save number of classes for model configuration
    num_classes = feature_extractor.num_classes
    x = np.concatenate(x_acc)
    y = np.concatenate(y_acc)
    # Process and save the datasets
    test_percentage = TRAINING_CONFIG["test_size"]
    validation_percentage = TRAINING_CONFIG["validation_size"]
    processor = ClassificationTrainingDatasetProcessor(x, y, test_percentage, validation_percentage, 7)
    dataset_info = processor.split_and_save(oversample)

    zero_value = feature_extractor.features_always_zero(dataset_info['x'])
    feature_extractor.save_feature_statistics(dataset_info['x'], f"_00_{len(x)}_feature_statistics.csv", comments=", ".join(zero_value))

    with open(os.path.join(DATA_DIR, 'class_info.json'), 'w') as f:
        json.dump({
            'num_classes': num_classes,
            'class_mapping': feature_extractor.movement_classes
        }, f)

    classes_frequency = ClassifierFeatureExtractor.describe_class_frequency(dataset_info['y'])
    print("Classes frequencies:", classes_frequency)
    print(f"Feature matrix shape: {dataset_info['x'].shape}")
    print(f"Class labels shape: {dataset_info['y'].shape}")
    print(f"Number of movement classes: {num_classes}")
    print(f"Training data shape: {dataset_info['x_train_shape']}")
    print(f"Validation data shape: {dataset_info['x_val_shape']}")
    print(f"Test data shape: {dataset_info['x_test_shape']}")


def name_datasets():
    data_loader = DataLoader(MEK_FILE)
    for root, _, files in os.walk(RAW_GAMEPLAY_LOGS_DIR):
        filtered_files = [file for file in files if file.endswith(".tsv")]
        with tqdm(total=len(filtered_files), desc="Renaming dataset: ") as t:
            for file in filtered_files:
                file_path = os.path.join(root, file)
                loaded_unit_actions, loaded_game_states, loaded_game_board = load_dataset_from_file(data_loader, file_path)
                if not loaded_unit_actions:
                    os.remove(file_path)
                    continue

                desc_text = file_path[-60:] + " " * (60 - len(file_path[-60:]))
                t.set_description("Renaming dataset: " + desc_text)
                t.update()

                unit_states = [unit_state for unit_state in loaded_game_states[0]]

                players = count_players(unit_states)
                units = count_units(unit_states)
                bv = count_bv(unit_states)
                quality = int(calculate_quality(loaded_unit_actions, loaded_game_states, loaded_game_board))

                new_name = create_file_name(bv, file_path, players, quality, loaded_unit_actions, units, prefix="dataset",
                                            extension="tsv")
                os.rename(file_path, new_name)


def read_first_line(file_path):
    with open(file_path, "r") as f:
        return f.readline()
