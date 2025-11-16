import React, { useState } from 'react';
import { Settings, Sliders, Layers, Zap, Wand2 } from 'lucide-react';
import SetupWizard from './SetupWizard';
import './ConfigPanel.css';

function ConfigPanel({ config, setConfig, disabled }) {
  const [showWizard, setShowWizard] = useState(false);

  const updateConfig = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleWizardComplete = (recommendedConfig) => {
    setConfig(recommendedConfig);
    setShowWizard(false);
  };

  return (
    <div className="config-panel fade-in">
      <div className="section-header">
        <h2>
          <Settings size={24} />
          Training Configuration
        </h2>
        <p>Adjust model architecture and training hyperparameters</p>
      </div>

      {disabled && (
        <div className="config-warning">
          <span className="badge-warning">Configuration locked during training</span>
        </div>
      )}

      {/* Training Parameters */}
      <div className="config-section">
        <h3>
          <Zap size={20} />
          Training Parameters
        </h3>
        <div className="config-grid">
          <div className="config-item">
            <label>
              Epochs
              <span className="config-hint">Number of complete passes through the dataset</span>
            </label>
            <input
              type="number"
              value={config.epochs}
              onChange={(e) => updateConfig('epochs', parseInt(e.target.value))}
              min={1}
              max={100}
              disabled={disabled}
            />
          </div>

          <div className="config-item">
            <label>
              Batch Size
              <span className="config-hint">Number of samples per training step</span>
            </label>
            <input
              type="number"
              value={config.batch_size}
              onChange={(e) => updateConfig('batch_size', parseInt(e.target.value))}
              min={1}
              max={128}
              disabled={disabled}
            />
          </div>

          <div className="config-item">
            <label>
              Learning Rate
              <span className="config-hint">Step size for weight updates</span>
            </label>
            <input
              type="number"
              value={config.learning_rate}
              onChange={(e) => updateConfig('learning_rate', parseFloat(e.target.value))}
              min={0.00001}
              max={0.01}
              step={0.0001}
              disabled={disabled}
            />
          </div>

          <div className="config-item">
            <label>
              Sequence Length
              <span className="config-hint">Maximum context length in tokens</span>
            </label>
            <input
              type="number"
              value={config.seq_length}
              onChange={(e) => updateConfig('seq_length', parseInt(e.target.value))}
              min={32}
              max={1024}
              step={32}
              disabled={disabled}
            />
          </div>
        </div>
      </div>

      {/* Model Architecture */}
      <div className="config-section">
        <h3>
          <Layers size={20} />
          Model Architecture
        </h3>
        <div className="config-grid">
          <div className="config-item">
            <label>
              Model Dimension (d_model)
              <span className="config-hint">Embedding and hidden layer size</span>
            </label>
            <input
              type="number"
              value={config.d_model}
              onChange={(e) => updateConfig('d_model', parseInt(e.target.value))}
              min={64}
              max={1024}
              step={64}
              disabled={disabled}
            />
          </div>

          <div className="config-item">
            <label>
              Number of Heads
              <span className="config-hint">Attention heads per layer</span>
            </label>
            <input
              type="number"
              value={config.num_heads}
              onChange={(e) => updateConfig('num_heads', parseInt(e.target.value))}
              min={1}
              max={16}
              disabled={disabled}
            />
          </div>

          <div className="config-item">
            <label>
              Number of Layers
              <span className="config-hint">Transformer blocks to stack</span>
            </label>
            <input
              type="number"
              value={config.num_layers}
              onChange={(e) => updateConfig('num_layers', parseInt(e.target.value))}
              min={1}
              max={24}
              disabled={disabled}
            />
          </div>

          <div className="config-item">
            <label>
              Feed-Forward Dimension
              <span className="config-hint">Inner dimension of FFN layers</span>
            </label>
            <input
              type="number"
              value={config.d_ff}
              onChange={(e) => updateConfig('d_ff', parseInt(e.target.value))}
              min={256}
              max={4096}
              step={256}
              disabled={disabled}
            />
          </div>
        </div>
      </div>

      {/* Presets */}
      <div className="config-section">
        <div className="presets-header">
          <h3>
            <Sliders size={20} />
            Quick Presets
          </h3>
          <button
            className="btn-wizard"
            onClick={() => setShowWizard(true)}
            disabled={disabled}
          >
            <Wand2 size={18} />
            Setup Wizard
          </button>
        </div>
        <div className="preset-buttons">
          <button
            onClick={() => setConfig({
              epochs: 5,
              batch_size: 32,
              learning_rate: 0.0003,
              seq_length: 64,
              d_model: 128,
              num_heads: 4,
              num_layers: 4,
              d_ff: 512
            })}
            disabled={disabled}
            className="btn-secondary"
          >
            Tiny Model
            <span className="preset-desc">Fast training, ~1M params</span>
          </button>

          <button
            onClick={() => setConfig({
              epochs: 10,
              batch_size: 32,
              learning_rate: 0.0003,
              seq_length: 128,
              d_model: 256,
              num_heads: 8,
              num_layers: 6,
              d_ff: 1024
            })}
            disabled={disabled}
            className="btn-secondary"
          >
            Small Model
            <span className="preset-desc">Balanced, ~10M params</span>
          </button>

          <button
            onClick={() => setConfig({
              epochs: 20,
              batch_size: 16,
              learning_rate: 0.0002,
              seq_length: 256,
              d_model: 512,
              num_heads: 8,
              num_layers: 8,
              d_ff: 2048
            })}
            disabled={disabled}
            className="btn-secondary"
          >
            Medium Model
            <span className="preset-desc">Better quality, ~50M params</span>
          </button>

          <button
            onClick={() => setConfig({
              epochs: 30,
              batch_size: 8,
              learning_rate: 0.00015,
              seq_length: 512,
              d_model: 768,
              num_heads: 12,
              num_layers: 12,
              d_ff: 3072
            })}
            disabled={disabled}
            className="btn-secondary"
          >
            Large Model
            <span className="preset-desc">High quality, ~100M params</span>
          </button>

          <button
            onClick={() => setConfig({
              epochs: 40,
              batch_size: 4,
              learning_rate: 0.0001,
              seq_length: 1024,
              d_model: 1024,
              num_heads: 16,
              num_layers: 16,
              d_ff: 4096
            })}
            disabled={disabled}
            className="btn-secondary"
          >
            XL Model
            <span className="preset-desc">Maximum quality, ~300M params</span>
          </button>

          <button
            onClick={() => setConfig({
              epochs: 3,
              batch_size: 64,
              learning_rate: 0.0005,
              seq_length: 32,
              d_model: 64,
              num_heads: 2,
              num_layers: 2,
              d_ff: 256
            })}
            disabled={disabled}
            className="btn-secondary btn-special"
          >
            Fast Experimentation
            <span className="preset-desc">Quick iterations, minimal compute</span>
          </button>
        </div>
      </div>

      {/* Estimated Parameters */}
      <div className="config-info">
        <h4>Estimated Model Size</h4>
        <p>
          This configuration will create a model with approximately{' '}
          <strong>
            {estimateParameters(config).toLocaleString()} parameters
          </strong>
        </p>
        <div className="config-tips">
          <h5>💡 Tips:</h5>
          <ul>
            <li>Start with smaller models if you have limited compute</li>
            <li>Increase batch size for faster training (if you have enough memory)</li>
            <li>Lower learning rate if training is unstable</li>
            <li>More layers = deeper understanding, but slower training</li>
          </ul>
        </div>
      </div>

      {/* Setup Wizard */}
      {showWizard && (
        <SetupWizard
          onComplete={handleWizardComplete}
          onClose={() => setShowWizard(false)}
        />
      )}
    </div>
  );
}

// Rough parameter count estimation
function estimateParameters(config) {
  const { d_model, num_layers, d_ff, vocab_size = 100 } = config;

  // Embedding layers
  const embeddings = vocab_size * d_model * 2; // token + position

  // Each transformer block has:
  // - Self-attention: 4 * d_model^2 (Q, K, V, output projection)
  // - Feed-forward: 2 * d_model * d_ff
  // - Layer norms: ~4 * d_model (small, can ignore)
  const perLayer = (4 * d_model * d_model) + (2 * d_model * d_ff);
  const allLayers = perLayer * num_layers;

  // Output projection
  const output = d_model * vocab_size;

  return Math.round(embeddings + allLayers + output);
}

export default ConfigPanel;
