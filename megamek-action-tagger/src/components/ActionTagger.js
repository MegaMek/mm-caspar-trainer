import React, { useState } from 'react';
import { useGameContext } from './GameContext';
import './ActionTagger.css';

// Movement Classification Types
const MOVEMENT_CLASSES = [
  "ADVANCE", "RETREAT", "FLANK", "HOLD_POSITION", "PURSUE",
  "EVADE", "SCOUT", "TAKE_COVER", "BREAKTHROUGH", "REPOSITION",
  "AMBUSH", "DEFEND_POINT", "SUPPORT_ALLY", "JUMP_ATTACK",
  "DEATH_FROM_ABOVE", "STRATEGIC_WITHDRAWAL"
];

// Quality ratings
const QUALITY_RATINGS = [
  "LOW_QUALITY", "HIGH_QUALITY", "IGNORE"
];

const ActionTagger = () => {
  const {
    filteredActions,
    currentActionIndex,
    setCurrentActionIndex,
    tags,
    setTags
  } = useGameContext();

  const [selectedClass, setSelectedClass] = useState(MOVEMENT_CLASSES[0]);
  const [selectedQuality, setSelectedQuality] = useState(QUALITY_RATINGS[1]);

  const handleCommit = () => {
    const currentAction = filteredActions[currentActionIndex];
    if (!currentAction) return;

    // Save the classification and quality rating
    setTags({
      ...tags,
      [currentActionIndex]: {
        entity: currentAction.entity_id,
        round: currentAction.round,
        classification: selectedClass,
        quality: selectedQuality
      }
    });

    // Move to the next action
    if (currentActionIndex < filteredActions.length - 1) {
      setCurrentActionIndex(currentActionIndex + 1);
    } else {
      alert("All actions have been tagged!");
    }
  };

  const exportTags = () => {
    const tagsJson = JSON.stringify(tags, null, 2);
    const blob = new Blob([tagsJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'action_tags.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Get current action
  const currentAction = filteredActions[currentActionIndex];

  // If we loaded existing tags, pre-select them
  React.useEffect(() => {
    if (currentAction && tags[currentAction.entity_id]) {
      setSelectedClass(tags[currentAction.entity_id].classification);
      setSelectedQuality(tags[currentAction.entity_id].quality);
    } else {
      setSelectedClass(MOVEMENT_CLASSES[0]);
      setSelectedQuality(QUALITY_RATINGS[1]);
    }
  }, [currentActionIndex, currentAction, tags]);

  return (
    <div className="controls-panel">
      <h2 className="section-title">Action Classification</h2>

      {currentAction ? (
        <div className="action-form">
          <div className="action-info">
            <h3>Current Action:</h3>
            <p><span className="label">Unit:</span> {currentAction.chassis} {currentAction.model}</p>
            <p><span className="label">Move:</span> ({currentAction.from_x},{currentAction.from_y}) → ({currentAction.to_x},{currentAction.to_y})</p>
            <p><span className="label">Hexes Moved:</span> {currentAction.hexes_moved}</p>
            <p><span className="label">Jumping:</span> {currentAction.jumping ? 'Yes' : 'No'}</p>
            <p><span className="label">Type:</span> {currentAction.type || 'BipedMek'}</p>
            <p><span className="label">Team:</span> {currentAction.team_id}</p>
            <p><span className="label">Is Bot:</span> {currentAction.is_bot}</p>
          </div>

          <div className="form-group">
            <label className="input-label">Movement Classification:</label>
            <select
              className="select-input"
              value={selectedClass}
              onChange={(e) => setSelectedClass(e.target.value)}
            >
              {MOVEMENT_CLASSES.map(classType => (
                <option key={classType} value={classType}>{classType}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="input-label">Quality Rating:</label>
            <select
              className="select-input"
              value={selectedQuality}
              onChange={(e) => setSelectedQuality(e.target.value)}
            >
              {QUALITY_RATINGS.map(rating => (
                <option key={rating} value={rating}>{rating}</option>
              ))}
            </select>
          </div>

          <div className="button-container">
            <button
              className="commit-button"
              onClick={handleCommit}
            >
              Commit & Next
            </button>

            <div className="counter">
              {currentActionIndex + 1} of {filteredActions.length}
            </div>
          </div>

          <div className="navigation-buttons">
            <button
              className="nav-button"
              onClick={() => setCurrentActionIndex(Math.max(0, currentActionIndex - 1))}
              disabled={currentActionIndex === 0}
            >
              Previous
            </button>

            <button
              className="nav-button"
              onClick={() => setCurrentActionIndex(Math.min(filteredActions.length - 1, currentActionIndex + 1))}
              disabled={currentActionIndex === filteredActions.length - 1}
            >
              Next
            </button>
          </div>
        </div>
      ) : (
        <p>No actions available to tag.</p>
      )}

      <div className="export-container">
        <button
          className="export-button"
          onClick={exportTags}
        >
          Export Tags
        </button>
      </div>
    </div>
  );
};

export default ActionTagger;