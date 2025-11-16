import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, TrendingDown, Zap, Clock } from 'lucide-react';
import './TrainingDashboard.css';

function TrainingDashboard({ trainingData, isTraining }) {
  // Get latest metrics
  const latest = trainingData[trainingData.length - 1];

  // Estimate time remaining (seconds) for current epoch based on step_time and remaining batches
  const computeEtaSeconds = (info) => {
    if (!info) return null;
    const stepTime = info.step_time;
    const totalBatches = info.total_batches;
    const batch = info.batch;
    if (!stepTime || !totalBatches || batch == null) return null;
    const remaining = Math.max(0, totalBatches - batch);
    return remaining * stepTime;
  };

  const etaSeconds = computeEtaSeconds(latest);

  const formatDuration = (secs) => {
    if (secs == null || isNaN(secs)) return 'N/A';
    const s = Math.max(0, Math.round(secs));
    const hours = Math.floor(s / 3600);
    const minutes = Math.floor((s % 3600) / 60);
    const seconds = s % 60;
    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
  };

  // Prepare chart data (sample every N points for performance)
  const sampleRate = Math.max(1, Math.floor(trainingData.length / 100));
  const chartData = trainingData.filter((_, idx) => idx % sampleRate === 0);

  return (
    <div className="training-dashboard fade-in">
      <div className="section-header">
        <h2>
          <Activity size={24} />
          Training Dashboard
        </h2>
        {isTraining && (
          <span className="badge-success pulse">
            Training in Progress
          </span>
        )}
      </div>

      {/* Metrics Cards */}
      {latest && (
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-icon" style={{background: 'rgba(239, 68, 68, 0.2)'}}>
              <TrendingDown size={24} style={{color: '#ef4444'}} />
            </div>
            <div className="metric-content">
              <span className="metric-label">Current Loss</span>
              <span className="metric-value">{latest.loss?.toFixed(4) || 'N/A'}</span>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon" style={{background: 'rgba(245, 158, 11, 0.2)'}}>
              <Activity size={24} style={{color: '#f59e0b'}} />
            </div>
            <div className="metric-content">
              <span className="metric-label">Perplexity</span>
              <span className="metric-value">{latest.perplexity?.toFixed(2) || 'N/A'}</span>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon" style={{background: 'rgba(16, 185, 129, 0.2)'}}>
              <Zap size={24} style={{color: '#10b981'}} />
            </div>
            <div className="metric-content">
              <span className="metric-label">Learning Rate</span>
              <span className="metric-value">{latest.learning_rate?.toFixed(6) || 'N/A'}</span>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon" style={{background: 'rgba(59, 130, 246, 0.2)'}}>
              <Clock size={24} style={{color: '#3b82f6'}} />
            </div>
            <div className="metric-content">
              <span className="metric-label">Tokens/sec</span>
              <span className="metric-value">{latest.tokens_per_sec?.toFixed(0) || 'N/A'}</span>
            </div>
          </div>
        </div>
      )}

      {/* Loss Chart */}
      {chartData.length > 0 ? (
        <div className="chart-container">
          <h3>Training Loss</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis
                dataKey="step"
                stroke="rgba(255,255,255,0.6)"
                label={{ value: 'Steps', position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.6)' }}
              />
              <YAxis
                stroke="rgba(255,255,255,0.6)"
                label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.6)' }}
              />
              <Tooltip
                contentStyle={{
                  background: 'rgba(22, 33, 62, 0.95)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  color: 'white'
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#ef4444"
                strokeWidth={2}
                dot={false}
                name="Loss"
              />
              <Line
                type="monotone"
                dataKey="avg_loss"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                name="Average Loss"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="no-data">
          <Activity size={48} />
          <h3>No Training Data</h3>
          <p>Start training to see real-time metrics and charts</p>
        </div>
      )}

      {/* Perplexity Chart */}
      {chartData.length > 0 && (
        <div className="chart-container">
          <h3>Perplexity</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis
                dataKey="step"
                stroke="rgba(255,255,255,0.6)"
                label={{ value: 'Steps', position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.6)' }}
              />
              <YAxis
                stroke="rgba(255,255,255,0.6)"
                label={{ value: 'Perplexity', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.6)' }}
              />
              <Tooltip
                contentStyle={{
                  background: 'rgba(22, 33, 62, 0.95)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  color: 'white'
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="perplexity"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                name="Perplexity"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Training Progress */}
      {latest && (
        <div className="progress-info">
          <h3>Training Progress</h3>
          <div className="progress-details">
            <div className="progress-item">
              <span className="progress-label">Epoch:</span>
              <span className="progress-value">{latest.epoch}</span>
            </div>
            <div className="progress-item">
              <span className="progress-label">Batch:</span>
              <span className="progress-value">{latest.batch} / {latest.total_batches}</span>
            </div>
            <div className="progress-item">
              <span className="progress-label">Global Step:</span>
              <span className="progress-value">{latest.step}</span>
            </div>
            <div className="progress-item">
              <span className="progress-label">Step Time:</span>
              <span className="progress-value">{latest.step_time?.toFixed(3)}s</span>
            </div>

            <div className="progress-item">
              <span className="progress-label">ETA:</span>
              <span className="progress-value">{etaSeconds !== null ? formatDuration(etaSeconds) : 'N/A'}</span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="progress-bar-container">
            <div
              className="progress-bar"
              style={{width: `${(latest.batch / latest.total_batches) * 100}%`}}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default TrainingDashboard;
