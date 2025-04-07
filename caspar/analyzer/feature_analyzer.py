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

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns


class FeatureAnalyzer:
    """
    Analyzes features for redundancy and importance.
    Follows Single Responsibility Principle by focusing only on analysis.
    """

    def __init__(self, correlation_threshold: float = 0.85):
        """Initialize with threshold for correlation analysis."""
        self.correlation_threshold = correlation_threshold

    def analyze_feature_correlations(self, X: np.ndarray, feature_names: List[str]) -> Tuple[pd.DataFrame, List[Tuple]]:
        """
        Identify highly correlated features that may be redundant.

        Args:
            X: Feature matrix from your dataset
            feature_names: List of feature names

        Returns:
            Tuple of (correlation_matrix, high_correlation_pairs)
        """
        # Convert to DataFrame for correlation analysis
        df = pd.DataFrame(X, columns=feature_names)

        # Calculate correlation matrix
        corr_matrix = df.corr().abs()

        # Find highly correlated feature pairs (upper triangle only)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Get pairs with correlation above threshold
        high_corr_indices = np.where(upper > self.correlation_threshold)

        # Create list of high correlation pairs
        high_corr_pairs = [(
            feature_names[high_corr_indices[0][i]],
            feature_names[high_corr_indices[1][i]],
            corr_matrix.iloc[high_corr_indices[0][i], high_corr_indices[1][i]]
        ) for i in range(len(high_corr_indices[0]))]

        return corr_matrix, high_corr_pairs

    def calculate_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Calculate importance of each feature based on the model.

        Args:
            model: Trained machine learning model
            feature_names: Names of features

        Returns:
            Dictionary mapping feature names to importance scores
        """
        feature_importances = {}

        # For tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for idx, importance in enumerate(importances):
                if idx < len(feature_names):
                    feature_importances[feature_names[idx]] = importance

        # For linear models
        elif hasattr(model, 'coef_'):
            coefficients = model.coef_
            # Handle both binary and multi-class cases
            if len(coefficients.shape) == 1:
                for idx, coef in enumerate(coefficients):
                    if idx < len(feature_names):
                        feature_importances[feature_names[idx]] = abs(coef)
            else:
                # For multi-class, average the absolute coefficients
                for idx, coef in enumerate(np.mean(np.abs(coefficients), axis=0)):
                    if idx < len(feature_names):
                        feature_importances[feature_names[idx]] = coef

        return feature_importances

    def visualize_correlation_matrix(self, corr_matrix: pd.DataFrame, output_path: str = 'correlation_heatmap.png'):
        """
        Visualize correlation matrix as a heatmap.

        Args:
            corr_matrix: Correlation matrix as pandas DataFrame
            output_path: Path to save the visualization
        """
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Correlation heatmap saved to {output_path}")

    def display_correlation_table(self, corr_matrix: pd.DataFrame, top_n: int = 20):
        """
        Display the correlation matrix as a styled table.

        Args:
            corr_matrix: Correlation matrix as pandas DataFrame
            top_n: Number of top correlations to display
        """
        # Reshape correlation matrix to a long format for easier ranking
        corr_long = corr_matrix.unstack().reset_index()
        corr_long.columns = ['Feature 1', 'Feature 2', 'Correlation']

        # Remove self-correlations and duplicates
        corr_long = corr_long[corr_long['Feature 1'] != corr_long['Feature 2']]
        corr_long = corr_long[~corr_long[['Feature 1', 'Feature 2']].apply(frozenset, axis=1).duplicated()]

        # Sort by absolute correlation
        corr_long = corr_long.sort_values('Correlation', ascending=False)

        # Display top correlations
        print(f"\nTop {top_n} Feature Correlations:")
        display_df = corr_long.head(top_n)

        # Format correlation values using .loc to avoid SettingWithCopyWarning
        display_df.loc[:, 'Correlation'] = display_df['Correlation'].map('{:.4f}'.format)

        return display_df