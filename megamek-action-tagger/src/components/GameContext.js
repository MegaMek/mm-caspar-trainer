import React, { createContext, useContext, useState, useEffect } from 'react';
import { preprocessTags, filteredActions as filterActions, filteredActionsAndGameStates } from './PreprocessTags';

// Create the context
const GameContext = createContext();

// Custom hook to use the game context
export const useGameContext = () => {
  const context = useContext(GameContext);
  if (!context) {
    throw new Error('useGameContext must be used within a GameProvider');
  }
  return context;
};

// Provider component
export const GameProvider = ({ children }) => {
  // State for game data
  const [gameBoard, setGameBoard] = useState({ width: 20, height: 20, hexes: [] });
  const [unitActions, setUnitActions] = useState([]);
  const [gameStates, setGameStates] = useState([]);
  const [filteredActions, setFilteredActions] = useState([]);
  const [currentActionIndex, setCurrentActionIndex] = useState(0);
  const [tags, setTags] = useState({});
  const [notes, setNotes] = useState('');
  const [isDataLoaded, setIsDataLoaded] = useState(false);

  // Initialize with mock data if no data is loaded
  useEffect(() => {
    if (!isDataLoaded) {
      initializeMockData();
    }
  }, [isDataLoaded]);

  const randomHex = () => {
    const elevation = Math.floor(Math.random() * 5) - 1;
    const has_water = Math.random() > 0.95;
    const depth = has_water ? Math.random() * 3 + 1 : 0
    const has_woods = has_water ? false : Math.random() > 0.7;
    const is_heavy_woods = has_woods ? Math.random() > 0.6 : false;
    const has_pavement = has_water || has_woods ? false : Math.random() > 0.9;
    const has_building = has_pavement ? Math.random() > 0.5 : false;
    const building_elevation = has_building ? Math.floor(Math.random() * 5) : 0;
    const floor = (has_building ? building_elevation : 0) + elevation - depth;
    return {
      elevation,
      has_water,
      depth,
      has_woods,
      is_heavy_woods,
      has_pavement,
      has_building,
      building_elevation,
      floor
    }
  }

  const initializeMockData = () => {
    // Create a mock board
    const mockBoard = {
      width: 20,
      height: 20,
      hexes: Array(20).fill().map(() => Array(20).fill().map(() => (randomHex())))
    };
    setGameBoard(mockBoard);

    // Create mock unit actions
    const mockActions = Array(20).fill().map((_, i) => ({
      player_id: Math.floor(Math.random() * 4),
      entity_id: i + 1,
      chassis: ['Turkina', 'Mad Cat', 'Ryoken', 'Thor', 'Night Gyr'][Math.floor(Math.random() * 5)],
      model: ['Prime', 'A', 'B', 'C', 'D'][Math.floor(Math.random() * 5)],
      facing: Math.floor(Math.random() * 6),
      from_x: Math.floor(Math.random() * 20),
      from_y: Math.floor(Math.random() * 20),
      to_x: Math.floor(Math.random() * 20),
      to_y: Math.floor(Math.random() * 20),
      hexes_moved: Math.floor(Math.random() * 8),
      jumping: Math.random() > 0.8 ? 1 : 0,
      is_bot: Math.random() > 0.8 ? 1 : 0,
      type: ['BipedMek', 'QuadMek', 'Tank', 'Infantry', 'BattleArmor'][Math.floor(Math.random() * 5)]
    }));
    setUnitActions(mockActions);

    // Create mock game states
    const mockGameStates = mockActions.map(action => {
      return Array(Math.floor(Math.random() * 10) + 5).fill().map(() => ({
        round: 1,
        phase: 'MOVEMENT',
        player_id: action.player_id,
        entity_id: action.entity_id,
        chassis: action.chassis,
        model: action.model,
        x: Math.floor(Math.random() * 20),
        y: Math.floor(Math.random() * 20),
        facing: Math.floor(Math.random() * 6),
        armor_p: Math.random(),
        internal_p: Math.random(),
        heat_p: Math.random() * 0.5,
        team_id: action.player_id % 2 + 1
      }));
    });
    setGameStates(mockGameStates);

    // Filter actions
    const filtered = filterActions(mockActions);
    setFilteredActions(filtered);
    setIsDataLoaded(true);
  };

  // Function to load data from JSON
  const loadGameData = (jsonData) => {
    try {
      // Parse JSON if it's a string
      const data = typeof jsonData === 'string' ? JSON.parse(jsonData) : jsonData;

      // Set the game board
      if (data.gameBoard) {
        setGameBoard(data.gameBoard);
      }
      // Reset current action index
      setCurrentActionIndex(0);

      if (data.gameStates && data.unitActions) {
        const filteredData = filteredActionsAndGameStates(data.unitActions, data.gameStates);
        setUnitActions(filteredData.filteredActions);
        setGameStates(filteredData.filteredStates);

        // Load tags if available
        if (data.tags) {
          setTags(data.tags);
        } else {
          setTags(preprocessTags(filteredData.filteredActions, filteredData.filteredStates, data.gameBoard))
        }
      }

      setIsDataLoaded(true);
      return true;
    } catch (error) {
      console.error("Error loading game data:", error);
      return false;
    }
  };

  // Export all data to JSON
  const exportAllData = () => {
    return JSON.stringify({
      gameBoard,
      unitActions,
      gameStates,
      tags,
      notes
    }, null, 2);
  };

  // Value to be provided by the context
  const value = {
    gameBoard,
    unitActions,
    gameStates,
    filteredActions,
    currentActionIndex,
    tags,
    isDataLoaded,
    setCurrentActionIndex,
    setTags,
    notes,
    setNotes,
    loadGameData,
    exportAllData
  };

  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
};