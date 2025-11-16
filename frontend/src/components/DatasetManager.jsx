import React, { useState } from 'react';
import { Plus, Link, File, Trash2, Database, AlertCircle } from 'lucide-react';
import './DatasetManager.css';

function DatasetManager({ apiUrl, onUpdate, datasetInfo }) {
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const addText = async () => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/dataset/add-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source: 'manual' })
      });

      if (response.ok) {
        setText('');
        onUpdate();
      } else {
        alert('Failed to add text');
      }
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const addURL = async () => {
    if (!url.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/dataset/add-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });

      if (response.ok) {
        setUrl('');
        onUpdate();
      } else {
        const error = await response.json();
        alert(`Failed to add URL: ${error.detail}`);
      }
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const uploadFile = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`${apiUrl}/dataset/add-file`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setSelectedFile(null);
        // Reset file input
        document.getElementById('file-input').value = '';
        onUpdate();
        alert(`File uploaded successfully! Added ${result.chars.toLocaleString()} characters.`);
      } else {
        const error = await response.json();
        alert(`Failed to upload file: ${error.detail}`);
      }
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const addSampleData = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/dataset/add-sample`, {
        method: 'POST'
      });

      if (response.ok) {
        onUpdate();
      } else {
        alert('Failed to add sample data');
      }
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const resetDataset = async () => {
    if (!confirm('Are you sure you want to reset the dataset?')) return;

    setLoading(true);
    try {
      await fetch(`${apiUrl}/dataset/reset`, { method: 'POST' });
      onUpdate();
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dataset-manager fade-in">
      <div className="section-header">
        <h2>
          <Database size={24} />
          Dataset Management
        </h2>
        <p>Add training data from multiple sources</p>
      </div>

      {/* Add Text */}
      <div className="dataset-card">
        <h3>
          <Plus size={20} />
          Add Custom Text
        </h3>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your training text here..."
          rows={6}
        />
        <button
          onClick={addText}
          disabled={!text.trim() || loading}
          className="btn-primary"
        >
          {loading ? <div className="spinner" /> : <Plus size={18} />}
          Add Text
        </button>
      </div>

      {/* Add URL */}
      <div className="dataset-card">
        <h3>
          <Link size={20} />
          Load from URL
        </h3>
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/article"
        />
        <button
          onClick={addURL}
          disabled={!url.trim() || loading}
          className="btn-primary"
        >
          {loading ? <div className="spinner" /> : <Link size={18} />}
          Fetch URL
        </button>
      </div>

      {/* Upload File */}
      <div className="dataset-card">
        <h3>
          <File size={20} />
          Upload File
        </h3>
        <div className="file-upload-section">
          <input
            id="file-input"
            type="file"
            onChange={handleFileSelect}
            accept=".txt,.md,.csv,.json,.log,.py,.js,.html,.xml"
            style={{ display: 'none' }}
          />
          <label htmlFor="file-input" className="file-input-label">
            <File size={18} />
            {selectedFile ? selectedFile.name : 'Choose a file...'}
          </label>
          {selectedFile && (
            <div className="file-info">
              <span>{(selectedFile.size / 1024).toFixed(2)} KB</span>
            </div>
          )}
        </div>
        <button
          onClick={uploadFile}
          disabled={!selectedFile || loading}
          className="btn-primary"
        >
          {loading ? <div className="spinner" /> : <File size={18} />}
          Upload File
        </button>
      </div>

      {/* Quick Actions */}
      <div className="dataset-card">
        <h3>Quick Actions</h3>
        <div className="button-group">
          <button
            onClick={addSampleData}
            disabled={loading}
            className="btn-secondary"
          >
            Add Sample Data
          </button>
          <button
            onClick={resetDataset}
            disabled={loading}
            className="btn-danger"
          >
            <Trash2 size={18} />
            Reset Dataset
          </button>
        </div>
      </div>

      {/* Dataset Info */}
      {datasetInfo?.has_data ? (
        <div className="dataset-info">
          <h3>Dataset Summary</h3>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Sources:</span>
              <span className="info-value">{datasetInfo.num_sources}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Total Characters:</span>
              <span className="info-value">{datasetInfo.total_chars.toLocaleString()}</span>
            </div>
          </div>

          <div className="sources-list">
            <h4>Sources:</h4>
            {datasetInfo.sources.map((source, idx) => (
              <div key={idx} className="source-item">
                <File size={16} />
                <span className="source-name">{source.source}</span>
                <span className="source-length">{source.length.toLocaleString()} chars</span>
              </div>
            ))}
          </div>

          {datasetInfo.sample && (
            <div className="sample-preview">
              <h4>Preview:</h4>
              <pre>{datasetInfo.sample}</pre>
            </div>
          )}
        </div>
      ) : (
        <div className="dataset-empty">
          <AlertCircle size={48} />
          <h3>No Data Yet</h3>
          <p>Add some training data to get started</p>
        </div>
      )}
    </div>
  );
}

export default DatasetManager;
