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
import React, { useState, useEffect, useCallback } from 'react';
import { useGameContext } from './GameContext';
import './ActionTagger.css';

// Movement Classification Types
const MOVEMENT_CLASSES = ["OFFENSIVE", "DEFENSIVE", "HOLD_POSITION"];

// Quality ratings
const QUALITY_RATINGS = ["HIGH_QUALITY", "LOW_QUALITY", "IGNORE"];

const ActionTagger = () => {
  const {
    unitActions,
    currentActionIndex,
    setCurrentActionIndex,
    tags,
    setTags,
    notes,
    setNotes,
    exportAllData,
    qualityIndex,
    loadGameData
  } = useGameContext();

  const [selectedClass, setSelectedClass] = useState(MOVEMENT_CLASSES[0]);
  const [selectedQuality, setSelectedQuality] = useState(QUALITY_RATINGS[0]);

  // Function to handle file selection for import - wrapped in useCallback
  const handleFileImport = useCallback(() => {
    // Create and trigger file input
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json';
    fileInput.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const jsonData = event.target.result;
          loadGameData(jsonData);
        } catch (error) {
          console.error("Error reading JSON:", error);
          alert('Error importing data');
        }
      };
      reader.readAsText(file);
    };
    fileInput.click();
  }, [loadGameData]);

  const handleCommit = useCallback(() => {
    const currentAction = unitActions[currentActionIndex];
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
    if (currentActionIndex < unitActions.length - 1) {
      setCurrentActionIndex(currentActionIndex + 1);
    } else {
      alert("All actions have been tagged!");
    }
  }, [currentActionIndex, unitActions, selectedClass, selectedQuality, setCurrentActionIndex, setTags, tags]);

  const getDatasetName = () => {
    // Get current timestamp in YYYYMMDD_HHMMSS format
    const now = new Date();
    const timestamp = now.toISOString()
      .replace(/[-:T.Z]/g, m => m === 'T' ? '_' : '')
      .substring(0, 15);

    // Count number of tags
    const tagCount = Object.keys(tags).length.toString().padStart(3, '0');

    // Generate random quality index between 0-100, padded to 3 digits
    const quality = qualityIndex.toString().padStart(3, '0');

    // Create simple hash from unitActions and tags
    const dataToHash = JSON.stringify({unitActions, tags});
    const simpleHash = Array.from(dataToHash)
      .reduce((hash, char) => ((hash << 5) - hash) + char.charCodeAt(0), 0)
      .toString(16).substring(0, 8);

    return `mm_tagged_dataset_${timestamp}_${tagCount}_${quality}_${simpleHash}.json`;
  }

  const handleExportData = useCallback((e) => {
    if (e) {
      e.preventDefault();
    }

    const jsonData = exportAllData();
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = getDatasetName();

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [exportAllData]);

  // Handle navigation - wrapped in useCallback
  const goPrevious = useCallback(() => {
    if (currentActionIndex > 0) {
      setCurrentActionIndex(currentActionIndex - 1);
    }
  }, [currentActionIndex, setCurrentActionIndex]);

  const goNext = useCallback(() => {
    if (currentActionIndex < unitActions.length - 1) {
      setCurrentActionIndex(currentActionIndex + 1);
    }
  }, [currentActionIndex, unitActions.length, setCurrentActionIndex]);

  // Get current action
  const currentAction = unitActions[currentActionIndex];

  // Load the tag for the current action when navigating
  useEffect(() => {
    if (currentAction) {
      // Look for tags by index first
      if (tags[currentActionIndex]) {
        setSelectedClass(tags[currentActionIndex].classification);
        setSelectedQuality(tags[currentActionIndex].quality);
      }
      // Fall back to default values if no tag exists
      else {
        setSelectedClass(MOVEMENT_CLASSES[0]);
        setSelectedQuality(QUALITY_RATINGS[1]);
      }
    }
  }, [currentActionIndex, currentAction, tags]);

  // Keyboard shortcut handler
  const handleKeyDown = useCallback((e) => {
    // Don't trigger shortcuts when typing in input fields
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
      return;
    }

    // Movement classification shortcuts (1-9)
    const num = parseInt(e.key, 10);
    if (!isNaN(num) && num >= 1 && num <= Math.min(9, MOVEMENT_CLASSES.length)) {
      setSelectedClass(MOVEMENT_CLASSES[num - 1]);
      e.preventDefault();
    }

    // Quality rating shortcuts
    if (e.key === 'q') {
      setSelectedQuality('HIGH_QUALITY');
      e.preventDefault();
    } else if (e.key === 'w') {
      setSelectedQuality('LOW_QUALITY');
      e.preventDefault();
    } else if (e.key === 'e') {
      setSelectedQuality('IGNORE');
      e.preventDefault();
    }

    // Commit and navigation shortcuts
    if (e.key === 'p') {
      handleCommit();
      e.preventDefault();
    } else if (e.key === '[') {
      goPrevious();
      e.preventDefault();
    } else if (e.key === ']') {
      goNext();
      e.preventDefault();
    }

    // Export/Import shortcuts
    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const cmdKey = isMac ? e.metaKey : e.ctrlKey;

    if (cmdKey && e.key === 's') {
      handleExportData(e);
      e.preventDefault();
    } else if (cmdKey && e.key === 'o') {
      handleFileImport();
      e.preventDefault();
    }
  }, [
    setSelectedClass,
    setSelectedQuality,
    handleCommit,
    goPrevious,
    goNext,
    handleExportData,
    handleFileImport
  ]);

  // Set up keyboard event listener
  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  return (
    <div className="controls-panel">
      <h2 className="section-title">Action Classification ({qualityIndex})</h2>

      {currentAction ? (
        <div className="action-form">
          <div className="action-info-container">
            <div className="action-info-left">
              <h3>Current Action: <span className="label">Team</span> {currentAction.team_id}</h3>
              <p><span className="label">Unit:</span>({currentAction.entity_id}) {currentAction.chassis} {currentAction.model} ({currentAction.role})</p>
              <p><span className="label">Type:</span> {currentAction.type || 'BipedMek'} <span className="label">BV:</span> {currentAction.bv}</p>
              <p><span className="label">Health:</span> {currentAction.armor} / {currentAction.internal} ({((currentAction.armor_p + currentAction.internal_p)/ 2 * 100).toFixed(1)}%) </p>
              <p><span className="label">Move:</span> ({currentAction.from_x+1},{currentAction.from_y+1}) → ({currentAction.to_x+1},{currentAction.to_y+1})</p>
              <p><span className="label">Hexes Moved:</span> {currentAction.hexes_moved} <span className="label">Jumping:</span> {currentAction.jumping ? 'Yes' : 'No'} <span className="label">Range:</span> {currentAction.max_range}</p>
            </div>
            <div className="action-info-right">
              <h3>Notes:</h3>
              <textarea
                className="notes-textarea"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add your notes about this movement here..."
                rows={6}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="input-label">Movement Classification: (1-9 keys)</label>
            <select
              className="select-input"
              value={selectedClass}
              onChange={(e) => setSelectedClass(e.target.value)}
            >
              {MOVEMENT_CLASSES.map((classType, index) => (
                <option key={classType} value={classType}>
                  {index + 1 <= 9 ? `${index + 1}. ` : ''}{classType}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="input-label">Quality Rating: (q=high, w=low, e=ignore)</label>
            <select
              className="select-input"
              value={selectedQuality}
              onChange={(e) => setSelectedQuality(e.target.value)}
            >
              {QUALITY_RATINGS.map(rating => (
                <option key={rating} value={rating}>
                  {rating === 'HIGH_QUALITY' ? 'q. ' :
                   rating === 'LOW_QUALITY' ? 'w. ' :
                   rating === 'IGNORE' ? 'e. ' : ''}{rating}
                </option>
              ))}
            </select>
          </div>

          <div className="button-container">
            <button
              className="commit-button"
              onClick={handleCommit}
              title="P"
            >
              Commit & Next
            </button>
            <button
              className="nav-button"
              onClick={goPrevious}
              disabled={currentActionIndex === 0}
              title="["
            >
              Previous
            </button>

            <button
              className="nav-button"
              onClick={goNext}
              disabled={currentActionIndex === unitActions.length - 1}
              title="]"
            >
              Next
            </button>
            <div className="counter">
              {currentActionIndex + 1} of {unitActions.length}
            </div>
          </div>

          <div className="shortcuts-info">
            <p><strong>Keyboard Shortcuts:</strong></p>
            <p>1-6: Select movement class</p>
            <p>q/w/e: High/Low/Ignore quality</p>
            <p>p: Commit & Next</p>
            <p>[/]: Previous/Next</p>
            <p>{navigator.platform.toUpperCase().indexOf('MAC') >= 0 ? '⌘+S' : 'Ctrl+S'}: Export Data</p>
            <p>{navigator.platform.toUpperCase().indexOf('MAC') >= 0 ? '⌘+O' : 'Ctrl+O'}: Import Data</p>
          </div>
        </div>
      ) : (
        <p>No actions available to tag.</p>
      )}

      <div className="export-container">
        <button
          className="import-button"
          onClick={handleFileImport}
          title={navigator.platform.toUpperCase().indexOf('MAC') >= 0 ? '⌘+O' : 'Ctrl+O'}
        >
          Import Data
        </button>
        <button
          className="export-button"
          onClick={handleExportData}
          title={navigator.platform.toUpperCase().indexOf('MAC') >= 0 ? '⌘+S' : 'Ctrl+S'}
        >
          Export Tagged Data
        </button>

      </div>
    </div>
  );
};

export default ActionTagger;