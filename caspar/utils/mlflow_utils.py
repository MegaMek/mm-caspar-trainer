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
import mlflow
from typing import Dict, Any, Optional


def setup_mlflow(tracking_uri: Optional[str] = None, experiment_name: str = "caspar-model"):
    """
    Set up MLflow tracking.
    
    Args:
        tracking_uri: URI for MLflow tracking server
        experiment_name: Name for MLflow experiment
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_system_metrics_sampling_interval(10)
    mlflow.set_experiment(experiment_name)
