import React from 'react';
import { useGameContext } from './GameContext';

const BoardPanel = () => {
  const { gameBoard, gameStates, unitActions, currentActionIndex } = useGameContext();

  // Get current action and related game state
  const currentAction = unitActions[currentActionIndex];
  const currentGameState = currentAction
    ? gameStates[currentActionIndex]
    : null;

  const getTerrainColor = (hex) => {
    if (!hex) return '#eee';

    let baseColor;

    if (hex.has_water) {
      // Blue tint for water
      const intensity = Math.floor(150 - hex.depth * 30);
      baseColor = `rgb(0, ${intensity}, 255)`;
    } else if (hex.has_woods) {
      // Green for woods
      const intensity = hex.is_heavy_woods ? 80 : 120;
      baseColor = `rgb(0, ${intensity}, 0)`;
    } else if (hex.has_pavement) {
      // Grey for pavement
      const elevationFactor = (hex.floor + 4) / 14; // Normalize -1 to 4 range to 0-1
      const intensity = Math.floor(180 * elevationFactor + 60);
      baseColor = `rgb( ${intensity}, ${intensity}, ${intensity})`;
    } else {
      // Brown scale for terrain levels 197, 227, 172
      const elevationFactor = (hex.floor + 2) / 16; // Normalize -4 to 10 range to 0-1
      const intensity = Math.floor(300 * elevationFactor);
      const red = Math.min(256, intensity + 147);
      const green = Math.min(intensity + 177, 256);
      const blue = Math.min(intensity + 122, 256);
      baseColor = `rgb(${red}, ${green}, ${blue})`;
    }

    return baseColor;
  };

  const buildings = ['🏢', '🏬', '🏭'];

  const getBuilding = (hex, x , y) => {
    if (!hex) return null;

    if (hex.has_building) {
      return buildings[x * y % buildings.length];
    }
    return null;
  }

  const getUnitIcon = (unit) => {
    const unitType = unit.type || 'BipedMek';

    switch(unitType) {
      case 'BipedMek':
      case 'LandAirMek':
      case 'TripodMek':
        return '🤖';
      case 'QuadMek':
        return '🐎';
      case 'HandheldWeapon':
        return '🔫';
      case 'ProtoMek':
        return '🦾';
      case 'Tank':
        return '🚜';
      case 'QuadVee':
        return '🚋';
      case 'SuperHeavyTank':
        return '🚛';
      case 'SupportTank':
        return '🚒';
      case 'BattleArmor':
        return '🥷';
      case 'Infantry':
        return '👥';
      case 'Warship':
      case 'SmallCraft':
      case 'AeroSpaceFighter':
        return '🚀';
      case 'FighterSquadron':
      case 'FixedWingSupport':
        return '✈️';
      case 'ConvFighter':
      case 'Aero':
        return '🛩️';
      case 'Dropship':
        return '🛸';
      case 'Jumpship':
        return '🛰️';
      case 'SpaceStation':
        return '💺';
      case 'VTOL':
        return '🚁';
      case 'SupportVTOL':
        return '🚁';
      case 'GunEmplacement':
        return '🗼';
      default:
        return '❓';
    }
  };

  const getUnitColor = (team_id) => {
    return team_id % 2 !== 0 ? '#33eeaa' : '#aa0123';
  };

   // Calculate arrow points for the SVG path
  const getArrowPath = () => {
    if (!currentAction) return '';

    // Cell size
    const cellSize = 21;

    // Calculate center points of cells
    const startX = currentAction.from_x * cellSize + cellSize/2 + 1;
    const startY = currentAction.from_y * cellSize + cellSize/2 + 1;
    const endX = currentAction.to_x * cellSize + cellSize/2 + 1;
    const endY = currentAction.to_y * cellSize + cellSize/2 + 1;

    // For long paths, shorten the arrow slightly to avoid overlapping with markers
    const dx = endX - startX;
    const dy = endY - startY;
    const length = Math.sqrt(dx * dx + dy * dy);

    // Adjust end point to stop short of destination cell center
    let adjustedEndX = endX;
    let adjustedEndY = endY;

    if (length > cellSize) {
      const shortenRatio = (length - cellSize/2) / length;
      adjustedEndX = startX + dx * shortenRatio;
      adjustedEndY = startY + dy * shortenRatio;
    }

    // Create path for line
    return `M ${startX} ${startY} L ${adjustedEndX} ${adjustedEndY}`;
  };


  // Calculate points for the arrowhead polygon
  const getArrowHeadPoints = () => {
    if (!currentAction) return '';

    const cellSize = 21;

    // Calculate center points
    const startX = currentAction.from_x * cellSize + cellSize/2 + 1;
    const startY = currentAction.from_y * cellSize + cellSize/2 + 1;
    const endX = currentAction.to_x * cellSize + cellSize/2 + 1;
    const endY = currentAction.to_y * cellSize + cellSize/2 + 1;

    // Calculate the angle for the arrowhead
    const angle = Math.atan2(endY - startY, endX - startX);

    // Size of arrowhead
    const arrowHeadSize = 8;

    // Calculate the three points of the arrowhead
    const tip_x = endX;
    const tip_y = endY;
    const left_x = endX - arrowHeadSize * Math.cos(angle - Math.PI/6);
    const left_y = endY - arrowHeadSize * Math.sin(angle - Math.PI/6);
    const right_x = endX - arrowHeadSize * Math.cos(angle + Math.PI/6);
    const right_y = endY - arrowHeadSize * Math.sin(angle + Math.PI/6);

    return `${tip_x},${tip_y} ${left_x},${left_y} ${right_x},${right_y}`;
  };

  return (
    <div className="board-panel">
      <div className="board-container">
        <div className="board-wrapper" style={{
          position: 'relative',
          width: `${gameBoard.height * 21}px`,
          height: `${gameBoard.width * 21}px`
        }}>
        <div
          className="game-board"
          style={{
              gridTemplateColumns: `repeat(${gameBoard.height}, 20px)`,
            gridTemplateRows: `repeat(${gameBoard.width}, 20px)`
          }}
        >
          {gameBoard.hexes.flatMap((column, y) =>
            column.map((hex, x) => (
              <div
                key={`${x}-${y}`}
                className="hex"
                style={{
                  backgroundColor: getTerrainColor(hex),
                }}
                title={`(${x},${y}) Elevation: ${hex?.floor ?? 0}`}
              >
                {/* Draw units on the map */}
                {currentGameState?.some(state => state.x === x && state.y === y) && (
                    currentGameState.find(state => state.x === x && state.y === y).team_id % 2 === 1 ? (
                  <div className="friendly-unit-icon">
                    {getUnitIcon(currentGameState.find(state => state.x === x && state.y === y))}
                  </div>
                    ) : (
                    <div className="enemy-unit-icon">
                    {getUnitIcon(currentGameState.find(state => state.x === x && state.y === y))}
                    </div>
                    ))
                }

                {/* Show current unit at its position */}
                {currentAction && currentAction.from_x === x && currentAction.from_y === y && (
                  <div
                    className="unit-icon"
                    style={{ border:  '4px solid ' + getUnitColor(currentAction.team_id) }}
                  >
                    {getUnitIcon(currentAction)}
                  </div>
                )}

                {/* Highlight current action path */}
                {currentAction && x === currentAction.from_x && y === currentAction.from_y && (
                  <div className="start-position"></div>
                )}
                {currentAction && x === currentAction.to_x && y === currentAction.to_y && (
                  <div className="end-position"></div>
                )}
                {!currentGameState?.some(state => state.x === x && state.y === y) &&
                    !((currentAction.to_x === x && currentAction.to_y === y) ||
                        (currentAction.from_x === x && currentAction.from_y === y)) &&
                    hex.has_building && (
                  <div>
                    {getBuilding(hex, x, y)}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
           {/* SVG overlay for drawing the arrow */}
          {currentAction && (
            <svg
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: `${gameBoard.height * 20}px`,
                height: `${gameBoard.width * 20}px`,
                pointerEvents: 'none',
                zIndex: 10
              }}
            >
              <path
                d={getArrowPath()}
                stroke="#ff4aff"
                strokeWidth="2"
                fill="none"
                strokeDasharray="5,5"
              />
              {/* Draw arrowhead manually instead of using markers for better control */}
              <polygon
                points={getArrowHeadPoints()}
                fill="#ff4aff"
              />
            </svg>
          )}
        </div>
      </div>
    </div>
  );
};

export default BoardPanel;