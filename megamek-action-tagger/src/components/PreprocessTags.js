// Utility function for preprocessing tags - isolated from component logic
// This function analyzes unit actions, game states, and the game board
// to automatically suggest appropriate movement classifications

const preprocessTags = (unitActions, gameStates, gameBoard) => {
  // Initialize empty tags object
  const tags = {};

  // Process each filtered action
  unitActions.forEach((action, index) => {
    // Default classification is ADVANCE
    let classification = "ADVANCE";
    let quality = "LOW_QUALITY";

    // Find all enemy positions from game states
    const friendlyTeamId = action.team_id;
    const enemyPositions = [];

    // Collect all enemy positions from all game states
    gameStates.forEach(stateGroup => {
      if (Array.isArray(stateGroup)) {
        stateGroup.forEach(state => {
          if (state.team_id !== friendlyTeamId) {
            enemyPositions.push(state);
          }
        });
      }
    });

    // If no movement, it's a HOLD_POSITION
    if (action.hexes_moved === 0 || (action.from_x === action.to_x && action.from_y === action.to_y)) {
      classification = "HOLD_POSITION";
    }
    // For all movements, analyze the positioning relative to enemies
    else {
      // Calculate the unit's weapon range (simplified; would be better to use actual weapon data)
      const unitMaxRange = action.max_range || 12; // Default to medium range if no data

      // Calculate distances to enemies before and after the move
      const enemyDistancesData = enemyPositions.map(enemy => {
        const distBefore = Math.sqrt(Math.pow(enemy.x - action.from_x, 2) + Math.pow(enemy.y - action.from_y, 2));
        const distAfter = Math.sqrt(Math.pow(enemy.x - action.to_x, 2) + Math.pow(enemy.y - action.to_y, 2));
        return {
          distBefore,
          distAfter,
          distChange: distBefore - distAfter, // Positive means moving toward enemy
          wasInRange: distBefore <= unitMaxRange,
          isInRange: distAfter <= unitMaxRange,
          isThreatening: distAfter <= enemy.max_range,
          wasThreatening: distBefore <= enemy.max_range
        };
      });

      // Sort by closest enemy after the move
      enemyDistancesData.sort((a, b) => a.distAfter - b.distAfter);

      // Check if there are any enemies in analysis
      if (enemyDistancesData.length > 0) {
        const closestEnemy = enemyDistancesData[0];
        const averageDistChange = enemyDistancesData.reduce((sum, data) => sum + data.distChange, 0) / enemyDistancesData.length;

        // OFFENSIVE: Moving toward enemy and putting them in attack range
        if (closestEnemy.distChange > 0 && closestEnemy.isInRange) {
          classification = "OFFENSIVE";
        }
        // DEFENSIVE: Moving away from enemy but keeping them in attack range
        else if (closestEnemy.distChange < 0 && (closestEnemy.isThreatening && closestEnemy.isInRange)) {
          classification = "DEFENSIVE";
        }
        // RETREAT: Moving away from enemies and leaving attack range
        else if (averageDistChange < 0 && !closestEnemy.isThreatening && !closestEnemy.isInRange) {
          classification = "RETREAT";
        }
        // ADVANCE: Moving toward enemies but not yet in attack range
        else if (averageDistChange > 0 && !closestEnemy.isInRange && !closestEnemy.isThreatening) {
          classification = "ADVANCE";
        // REPOSITION: Short movements that don't change engagement status significantly
        }

        if (Math.abs(averageDistChange) < 3 && action.hexes_moved < 4) {
          classification = "REPOSITION";
        }
      } else {
        // If no enemies detected, classify based on movement distance
        if (action.hexes_moved >= 4) {
          classification = "ADVANCE";
        } else {
          classification = "REPOSITION";
        }
      }

      // Handle jumps (often tactical repositioning)
      if (action.jumping === 1) {
        // Jumping is usually a reposition unless it's clearly offensive or retreating
        if (classification !== "OFFENSIVE" && classification !== "RETREAT") {
          classification = "REPOSITION";
        }
      }
    }

    // Assess move quality based on MP efficiency and terrain choice
    if (action.mp_used > action.max_mp * 0.75) {
      // Using most movement points is usually good
      quality = "HIGH_QUALITY";
    }

    // Add to tags object
    tags[index] = {
      entity: action.entity_id,
      round: action.round || 0,
      classification,
      quality
    };
  });

  return tags;
};

const FILTER_TYPES = ['AeroSpaceFighter', 'Infantry', 'FixedWingSupport', 'ConvFighter',
      'Dropship', 'EjectedCrew', 'MekWarrior', 'GunEmplacement', 'BattleArmor'];

const filteredActions = (unitActions) => unitActions.filter(action =>
    action.is_bot !== 1 &&
    !FILTER_TYPES.includes(action.type || 'BipedMek')
  );

const filteredActionsAndGameStates = (unitActions, gameStates) => {
  let filteredActions = [];
  let filteredStates = [];
  for (let i = 0; i < unitActions.length; i++) {
    const action = unitActions[i];
    const state = gameStates[i];

    if (action.is_bot === 1) {
      continue;
    }

    if (FILTER_TYPES.includes(action.type || 'BipedMek')) {
      continue;
    }

    filteredActions.push(action);
    filteredStates.push(state);
  }
  return {filteredActions, filteredStates};
}

// Export the preprocessTags function for use outside the component
export { preprocessTags, filteredActions, filteredActionsAndGameStates };