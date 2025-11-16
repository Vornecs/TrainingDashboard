import React, { useState, useEffect } from 'react';
import { Brain, Database, Activity, MessageSquare, Settings } from 'lucide-react';
import DatasetManager from './components/DatasetManager';
import TrainingDashboard from './components/TrainingDashboard';
import ChatInterface from './components/ChatInterface';
import ConfigPanel from './components/ConfigPanel';
import './App.css';

const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

function App() {
  const [activeTab, setActiveTab] = useState('dataset');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingData, setTrainingData] = useState([]);
  const [websocket, setWebsocket] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [config, setConfig] = useState({
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.0003,
    seq_length: 128,
    d_model: 256,
    num_heads: 8,
    num_layers: 6,
    d_ff: 1024
  });

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setWebsocket(ws);
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'training_step') {
        setTrainingData(prev => [...prev.slice(-100), message.data]); // Keep last 100 steps
      } else if (message.type === 'training_epoch') {
        console.log('Epoch completed:', message.data);
      } else if (message.type === 'training_complete') {
        setIsTraining(false);
        alert('Training completed!');
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWebsocket(null);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  // Fetch dataset info
  const fetchDatasetInfo = async () => {
    try {
      const response = await fetch(`${API_URL}/dataset/info`);
      const data = await response.json();
      setDatasetInfo(data);
    } catch (error) {
      console.error('Error fetching dataset info:', error);
    }
  };

  useEffect(() => {
    fetchDatasetInfo();
    const interval = setInterval(fetchDatasetInfo, 5000);
    return () => clearInterval(interval);
  }, []);

  // Start training
  const startTraining = async () => {
    if (!datasetInfo?.has_data) {
      alert('Please add data to the dataset first!');
      return;
    }

    try {
      const response = await fetch(`${API_URL}/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (response.ok) {
        setIsTraining(true);
        setTrainingData([]);
        setActiveTab('training');
      } else {
        const error = await response.json();
        alert(`Failed to start training: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error starting training:', error);
      alert('Failed to start training');
    }
  };

  // Stop training
  const stopTraining = async () => {
    try {
      await fetch(`${API_URL}/training/stop`, { method: 'POST' });
      setIsTraining(false);
    } catch (error) {
      console.error('Error stopping training:', error);
    }
  };

  const tabs = [
    { id: 'dataset', name: 'Dataset', icon: Database },
    { id: 'config', name: 'Configuration', icon: Settings },
    { id: 'training', name: 'Training', icon: Activity },
    { id: 'chat', name: 'Chat', icon: MessageSquare }
  ];

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Brain size={32} />
            <h1>AI Training Platform</h1>
          </div>
          <div className="header-actions">
            {isTraining ? (
              <button onClick={stopTraining} className="btn-danger">
                Stop Training
              </button>
            ) : (
              <button
                onClick={startTraining}
                className="btn-success"
                disabled={!datasetInfo?.has_data}
              >
                Start Training
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Sidebar Navigation */}
        <nav className="sidebar">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon size={20} />
                <span>{tab.name}</span>
              </button>
            );
          })}
        </nav>

        {/* Content Area */}
        <div className="content">
          {activeTab === 'dataset' && (
            <DatasetManager
              apiUrl={API_URL}
              onUpdate={fetchDatasetInfo}
              datasetInfo={datasetInfo}
            />
          )}
          {activeTab === 'config' && (
            <ConfigPanel
              config={config}
              setConfig={setConfig}
              disabled={isTraining}
            />
          )}
          {activeTab === 'training' && (
            <TrainingDashboard
              trainingData={trainingData}
              isTraining={isTraining}
            />
          )}
          {activeTab === 'chat' && (
            <ChatInterface apiUrl={API_URL} />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>
          Training a Transformer from scratch • Built with PyTorch & React
          {datasetInfo?.has_data && (
            <span className="dataset-badge">
              {datasetInfo.num_sources} source(s) • {datasetInfo.total_chars.toLocaleString()} chars
            </span>
          )}
        </p>
      </footer>
    </div>
  );
}

export default App;
