import React, { useRef, useState } from 'react';
import { useGameContext } from './GameContext';

const DataImporter = () => {
  const { loadGameData, exportAllData } = useGameContext();
  const [importMessage, setImportMessage] = useState('');
  const [messageType, setMessageType] = useState(''); // 'success' or 'error'
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setImportMessage('');
    setMessageType('');

    // Check if it's a JSON file
    if (!file.name.endsWith('.json')) {
      setImportMessage('Please select a JSON file');
      setMessageType('error');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const jsonData = e.target.result;
        const success = loadGameData(jsonData);

        if (success) {
          setImportMessage('Data imported successfully!');
          setMessageType('success');
        } else {
          setImportMessage('Failed to import data. Invalid format.');
          setMessageType('error');
        }
      } catch (error) {
        console.error("Error reading JSON:", error);
        setImportMessage('Error reading JSON file');
        setMessageType('error');
      }
    };

    reader.onerror = () => {
      setImportMessage('Failed to read file');
      setMessageType('error');
    };

    reader.readAsText(file);

    // Reset the file input
    event.target.value = null;
  };

  const handleImportClick = () => {
    fileInputRef.current.click();
  };

  const handleExportClick = () => {
    const jsonData = exportAllData();
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'megamek_data.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="data-importer">
      <h2 className="section-title">Import/Export Data</h2>

      <div className="importer-buttons">
        <input
          type="file"
          accept=".json"
          ref={fileInputRef}
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        <button
          className="import-button"
          onClick={handleImportClick}
        >
          Import Game Data
        </button>
        <button
          className="export-button"
          onClick={handleExportClick}
        >
          Export Game Data
        </button>
      </div>

      {importMessage && (
        <div className={`message ${messageType}`}>
          {importMessage}
        </div>
      )}
    </div>
  );
};

export default DataImporter;