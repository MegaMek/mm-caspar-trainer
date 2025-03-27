import React from 'react';
import { GameProvider } from './components/GameContext';
import BoardPanel from './components/BoardPanel';
import ActionTagger from './components/ActionTagger';
import DataImporter from './components/DataImporter';
import './App.css';

function App() {
  return (
    <div className="App">
      <GameProvider>
        <div className="main-container">
          <BoardPanel />
          <ActionTagger />
        </div>
      </GameProvider>
    </div>
  );
}

export default App;