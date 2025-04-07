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
import React from 'react';
import { useGameContext } from './GameContext';
import './HexBoardPanel.css';

const BoardPanel = () => {
  const { gameBoard, gameStates, unitActions, currentActionIndex } = useGameContext();

  // Get current action and related game state
  const currentAction = unitActions[currentActionIndex];
  const currentGameState = currentAction
    ? gameStates[currentActionIndex]
    : null;

  // Hex grid constants
  const hexSize = 18; // Radius of the hex
  const hexWidth = hexSize * 2;
  const hexHeight = Math.sqrt(3) / 2 * hexWidth; // Height of a hex

  // Calculate the center point for a hex at given coordinates
  const getHexCenter = (x, y) => {
    // For even-q offset coordinates with flat-topped hexes
    // Even columns are offset (shifted down)
    const xPos = x * (3/4 * hexWidth);
    let yPos = y * hexHeight;

    // Shift even columns down
    if (x % 2 !== 0) {
      yPos += hexHeight / 2;
    }

    return { x: xPos + hexSize, y: yPos + hexSize };
  };

  // Generate points for drawing a hexagon
  const getHexPoints = (centerX, centerY) => {
    let points = [];
    for (let i = 0; i < 6; i++) {
      const angleDeg = 60 * i;
      const angleRad = Math.PI / 180 * angleDeg;
      const x = centerX + hexSize * Math.cos(angleRad);
      const y = centerY + hexSize * Math.sin(angleRad);
      points.push(`${x},${y}`);
    }
    return points.join(' ');
  };


  // Generate points for drawing a hexagon
  const getGiantHexPoints = (centerX, centerY, radius) => {
    let points = [];
    for (let i = 0; i < 6; i++) {
      const angleDeg = 60 * i;
      const angleRad = Math.PI / 180 * angleDeg;
      const x = centerX + (hexSize * radius*2 * 0.7) * Math.cos(angleRad);
      const y = centerY + (hexSize * radius*2 * 0.7) * Math.sin(angleRad);
      points.push(`${x},${y}`);
      points.push(`${centerX},${centerY}`);
      points.push(`${x},${y}`);
    }
    return points.join(' ');
  };

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

  const getBuilding = (hex, x, y) => {
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

  // Calculate the path for movement arrow
  const getArrowPath = () => {
    if (!currentAction) return '';

    // Get center points of source and destination hexes
    const start = getHexCenter(currentAction.from_x, currentAction.from_y);
    const end = getHexCenter(currentAction.to_x, currentAction.to_y);

    // For long paths, shorten the arrow slightly to avoid overlapping with markers
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const length = Math.sqrt(dx * dx + dy * dy);

    // Adjust end point to stop short of destination cell center
    let adjustedEndX = end.x;
    let adjustedEndY = end.y;

    if (length > hexSize) {
      const shortenRatio = (length - hexSize/2) / length;
      adjustedEndX = start.x + dx * shortenRatio;
      adjustedEndY = start.y + dy * shortenRatio;
    }

    // Create path for line
    return `M ${start.x} ${start.y} L ${adjustedEndX} ${adjustedEndY}`;
  };

  /**
 * Generates points for an arrow indicating the unit's facing direction
 * @param {number} x - X coordinate of the hex
 * @param {number} y - Y coordinate of the hex
 * @param {number} facing - Direction (0=N, 1=NE, 2=SE, 3=S, 4=SW, 5=NW)
 * @param {number} hexSize - Size of hexagon
 * @returns {string} - SVG polygon points for the arrowhead
 */
const getFacing = (x, y, facing, hexSize) => {
  // Get center of the hex
  const center = getHexCenter(x, y);

  // Convert facing to radians (flat-topped hex)
  // 0 = North (90°), 1 = NE (30°), 2 = SE (330°), 3 = S (270°), 4 = SW (210°), 5 = NW (150°)
  const facingAngles = [
    Math.PI/2,    // 0: North (90°)
    Math.PI/6,    // 1: Northeast (30°)
    -Math.PI/6,   // 2: Southeast (330°)
    -Math.PI/2,   // 3: South (270°)
    -5*Math.PI/6, // 4: Southwest (210°)
    5*Math.PI/6   // 5: Northwest (150°)
  ];

  const angle = facingAngles[facing % 6];

  // Arrow tip position (1 hexSize away from center)
  const tipX = center.x + hexSize * Math.cos(angle);
  const tipY = center.y - hexSize * Math.sin(angle); // Subtract for y because SVG y-axis is inverted

  // Arrow width (perpendicular to direction)
  const arrowWidth = hexSize * 0.4;

  // Calculate perpendicular angle (90 degrees = PI/2 radians from facing direction)
  const perpAngle = angle + Math.PI/2;

  // Calculate arrow base points (perpendicular to direction)
  const baseX = tipX - 0.6 * hexSize * Math.cos(angle);
  const baseY = tipY + 0.6 * hexSize * Math.sin(angle); // Add for y because SVG y-axis is inverted

  // Calculate the two base corners of the arrowhead
  const leftX = baseX + arrowWidth/2 * Math.cos(perpAngle);
  const leftY = baseY - arrowWidth/2 * Math.sin(perpAngle);

  const rightX = baseX - arrowWidth/2 * Math.cos(perpAngle);
  const rightY = baseY + arrowWidth/2 * Math.sin(perpAngle);

  // Return polygon points
  return `${tipX},${tipY} ${leftX},${leftY} ${rightX},${rightY}`;
};

  // Calculate points for the arrowhead polygon
  const getArrowHeadPoints = () => {
    if (!currentAction) return '';

    // Get center points
    const start = getHexCenter(currentAction.from_x, currentAction.from_y);
    const end = getHexCenter(currentAction.to_x, currentAction.to_y);

    // Calculate the angle for the arrowhead
    const angle = Math.atan2(end.y - start.y, end.x - start.x);

    // Size of arrowhead
    const arrowHeadSize = 8;

    // Calculate the three points of the arrowhead
    const tip_x = end.x;
    const tip_y = end.y;
    const left_x = end.x - arrowHeadSize * Math.cos(angle - Math.PI/6);
    const left_y = end.y - arrowHeadSize * Math.sin(angle - Math.PI/6);
    const right_x = end.x - arrowHeadSize * Math.cos(angle + Math.PI/6);
    const right_y = end.y - arrowHeadSize * Math.sin(angle + Math.PI/6);

    return `${tip_x},${tip_y} ${left_x},${left_y} ${right_x},${right_y}`;
  };

  // Calculate board dimensions based on hex sizes
  const boardHeight = gameBoard.width * hexHeight + hexHeight/2;
  const boardWidth = gameBoard.height * (3/4 * hexWidth) + hexWidth/4;

  return (
    <div className="board-panel">
      <div className="board-container">
        <div className="board-wrapper" style={{
          position: 'relative',
            overflow: "auto",
          width: `${boardWidth}px`,
          height: `${boardHeight}px`
        }}>
          <svg
            className="game-board"
            width={boardWidth}
            height={boardHeight}
            style={{
              position: 'absolute',
              top: 0,
              left: 0
            }}
          >
            {/* Draw hex grid */}
            {gameBoard.hexes.flatMap((column, y) =>
              column.map((hex, x) => {
                const center = getHexCenter(x, y);
                const hexPoints = getHexPoints(center.x, center.y);

                return (
                  <g key={`${x}-${y}`}>
                    {/* Draw hex background */}
                    <polygon
                      points={hexPoints}
                      fill={getTerrainColor(hex)}
                      stroke="#333"
                      strokeWidth="0.5"
                    />

                    {/* Highlight start position */}
                    {currentAction && x === currentAction.from_x && y === currentAction.from_y && (
                      <circle
                        cx={center.x}
                        cy={center.y}
                        r={hexSize * 0.5}
                        fill="transparent"
                        stroke="#4ff"
                        strokeWidth="2"
                      />
                    )}

                    {/* Highlight end position */}
                    {currentAction && x === currentAction.to_x && y === currentAction.to_y && (
                      <>
                      <circle
                        cx={center.x}
                        cy={center.y}
                        r={hexSize * 0.5}
                        fill="transparent"
                        stroke="#ff4"
                        strokeWidth="2"
                      />
                      <polygon
                          points={getFacing(currentAction.to_x, currentAction.to_y, currentAction.facing, hexSize / 2)}
                          fill="#ffcc00"
                          stroke="#000000"
                          strokeWidth="5"
                      />
                    </>
                    )}

                    {/* Draw units on the map */}
                    {currentGameState?.some(state => state.x === x && state.y === y) && (
                      <foreignObject
                        x={center.x - hexSize/2}
                        y={center.y - hexSize/2}
                        width={hexSize}
                        height={hexSize}
                      >
                        <div
                          className={currentGameState.find(state => state.x === x && state.y === y).team_id === currentAction.team_id ? "unit-friendly" : "unit-enemy"}
                          style={{
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: `${hexSize * 0.8}px`,
                          }}
                        >
                          {getUnitIcon(currentGameState.find(state => state.x === x && state.y === y))}
                        </div>
                      </foreignObject>
                    )}

                    {/* Show current unit at its from position */}
                    {currentAction && currentAction.from_x === x && currentAction.from_y === y && (
                      <foreignObject
                        x={center.x - hexSize/2}
                        y={center.y - hexSize/2}
                        width={hexSize}
                        height={hexSize}
                      >
                        <div
                          style={{
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: `${hexSize * 0.8}px`,
                            textShadow: `0 0 3px ${getUnitColor(currentAction.team_id)}`
                          }}
                        >
                          {getUnitIcon(currentAction)}
                        </div>
                      </foreignObject>
                    )}

                    {/* Buildings */}
                    {!currentGameState?.some(state => state.x === x && state.y === y) &&
                      !((currentAction?.to_x === x && currentAction?.to_y === y) ||
                        (currentAction?.from_x === x && currentAction?.from_y === y)) &&
                      hex?.has_building && (
                      <foreignObject
                        x={center.x - hexSize/2}
                        y={center.y - hexSize/2}
                        width={hexSize}
                        height={hexSize}
                      >
                        <div
                          style={{
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: `${hexSize * 0.8}px`
                          }}
                        >
                          {getBuilding(hex, x, y)}
                        </div>
                      </foreignObject>
                    )}

                    {/* Coordinates for debugging */}
                    <text
                      x={center.x}
                      y={center.y + 3}
                      textAnchor="middle"
                      fontSize="6"
                      fill="#00000077"
                      style={{ pointerEvents: 'none' }}
                    >
                      {x+1},{y+1} ({hex.floor})
                    </text>
                  </g>
                );
              })
            )}

            {/* Draw movement arrow */}
            {currentAction && (
              <>
                <path
                  d={getArrowPath()}
                  stroke="#ff4aff"
                  strokeWidth="2"
                  fill="none"
                  strokeDasharray="5,5"
                />
                <polygon
                  points={getArrowHeadPoints()}
                  fill="#ff4aff"
                />
                <polygon
                  points={getGiantHexPoints(getHexCenter(currentAction.to_x, currentAction.to_y).x, getHexCenter(currentAction.to_x, currentAction.to_y).y, currentAction.max_range)}
                  fill="transparent"
                  stroke="#3300ff33"
                  strokeWidth="2"
                />
              </>
            )}
          </svg>
        </div>
      </div>
    </div>
  );
};

export default BoardPanel;