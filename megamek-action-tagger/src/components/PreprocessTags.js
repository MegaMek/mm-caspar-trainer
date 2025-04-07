/* Copyright (C) 2025-2025 The MegaMek Team. All Rights Reserved.
*
* This file is part of MM-Caspar-Trainer.
*
* MM-Caspar-Trainer is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License (GPL),
* version 3 or (at your option) any later version,
* as published by the Free Software Foundation.
*
* MM-Caspar-Trainer is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty
* of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU General Public License for more details.
*
* A copy of the GPL should have been included with this project;
* if not, see <https://www.gnu.org/licenses/>.
*
* NOTICE: The MegaMek organization is a non-profit group of volunteers
* creating free software for the BattleTech community.
*
* MechWarrior, BattleMech, `Mech and AeroTech are registered trademarks
* of The Topps Company, Inc. All Rights Reserved.
*
* Catalyst Game Labs and the Catalyst Game Labs logo are trademarks of
* InMediaRes Productions, LLC.
*/

/**
 * Checks if a unit is stationary
 * @param {Object} action - The unit action
 * @param {Object} _closestEnemy - Unused in this function
 * @returns {boolean} - True if unit didn't move
 */
const isStationary = (action, _closestEnemy) => {
  return action.hexes_moved === 0 ||
         (action.from_x === action.to_x && action.from_y === action.to_y);
};

/**
 * Checks if movement is offensive
 * @param {Object} action - The unit action
 * @param {Object} closestEnemy - Closest enemy data
 * @returns {boolean} - True if offensive movement
 */
const isOffensive = (action, closestEnemy) => {
  return action.hexes_moved > 2 && closestEnemy > 0;
};

/**
 * Checks if movement is defensive
 * @param {Object} action - The unit action
 * @param {Object} closestEnemy - Closest enemy data
 * @returns {boolean} - True if defensive movement
 */
const isDefensive = (action, closestEnemy) => {
  return action.hexes_moved > 2 && closestEnemy < 0;
};

/**
 * Checks if movement is a reposition
 * @param {Object} action - The unit action
 * @param {Object} _closestEnemy - Unused in this function
 * @returns {boolean} - True if reposition movement
 */
const isReposition = (action, _closestEnemy) => {
  return action.hexes_moved < 3;
};

/**
 * Priority order for checking behaviors
 */
const BEHAVIOR_PRIORITY = [
  "HOLD_POSITION",
  "REPOSITION",
  "OFFENSIVE",
  "DEFENSIVE",
];

/**
 * Definition of movement behaviors with their check functions
 */
const BEHAVIORS = {
  "HOLD_POSITION": isStationary,
  "OFFENSIVE": isOffensive,
  "DEFENSIVE": isDefensive,
  "REPOSITION": isReposition,
};

/**
 * Main function to classify movement using behavior configuration
 * @param {Object} action - The unit action
 * @param {Array} enemyDistancesData - Array of enemy distance data
 * @returns {string} - Movement classification
 */
const classifyMovement = (action, distanceDelta) => {
  // Default classification
  let behaviorType = "DEFENSIVE";
  // Check behaviors in priority order
  for (const behavior of BEHAVIOR_PRIORITY) {
    const checkFunction = BEHAVIORS[behavior];

    if (checkFunction && checkFunction(action, distanceDelta)) {
      behaviorType = behavior;
      break;
    }
  }

  return behaviorType;
};

/**
 * Extract enemy positions from game states
 * @param {Array} gameStates - Game state data
 * @param {number} friendlyTeamId - ID of friendly team
 * @returns {Array} - Array of enemy positions
 */
const extractEnemyPositions = (gameStates, friendlyTeamId) => {
  const enemyPositions = [];

  gameStates.forEach(stateGroup => {
    if (Array.isArray(stateGroup)) {
      stateGroup.forEach(state => {
        if (state.team_id !== friendlyTeamId) {
          enemyPositions.push(state);
        }
      });
    }
  });

  return enemyPositions;
};

/**
 * Calculate distance between two points
 * @param {number} x1 - First point x
 * @param {number} y1 - First point y
 * @param {number} x2 - Second point x
 * @param {number} y2 - Second point y
 * @returns {number} - Euclidean distance
 */
const calculateDistance = (x1, y1, x2, y2) => {
  return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
};

/**
 * Assess move quality
 * @param {Object} action - The unit action
 * @returns {string} - Quality assessment
 */
const assessMoveQuality = (action) => {
  if (action.from_x === -1 && action.from_y === -1) {
    return "IGNORE";
  } else if (action.mp_used > action.max_mp * 0.75) {
    return "HIGH_QUALITY";
  }
  return "LOW_QUALITY";
};

/**
 * Create a tag object
 * @param {Object} action - The unit action
 * @param {string} classification - Movement classification
 * @param {string} quality - Quality assessment
 * @returns {Object} - Tag object
 */
const createTag = (action, classification, quality) => {
  return {
    entity: action.entity_id,
    round: action.round || 0,
    classification,
    quality
  };
};

/**
 * Main function to preprocess tags
 * @param {Array} unitActions - Unit actions
 * @param {Array} gameStates - Game states
 * @param {Object} _gameBoard - Game board (unused)
 * @returns {Object} - Tags object
 */
const preprocessTags = (unitActions, gameStates, _gameBoard) => {
  const tags = {};

  unitActions.forEach((action, index) => {
    const friendlyTeamId = action.team_id;
    const enemyPositions = extractEnemyPositions(gameStates, friendlyTeamId);
    const closestEnemyStart = distanceFrom(action.from_x, action.from_y, enemyPositions);
    const closestEnemyEnd = distanceFrom(action.to_x, action.to_y, enemyPositions);
    const distanceDelta = closestEnemyStart - closestEnemyEnd
    const classification = classifyMovement(action, distanceDelta);
    const quality = assessMoveQuality(action);

    tags[index] = createTag(action, classification, quality);
  });

  return tags;
};

const distanceFrom = (x, y, units) => {
  let minDist = 9999999999;
  for (let unit of units) {
    let dist = calculateDistance(unit.x, unit.y, x, y);
    minDist = Math.min(dist, minDist)
  }

  return minDist;
}


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