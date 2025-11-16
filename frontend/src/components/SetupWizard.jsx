import React, { useState } from 'react';
import { Wand2, ChevronRight, ChevronLeft, Check, X } from 'lucide-react';
import './SetupWizard.css';

function SetupWizard({ onComplete, onClose }) {
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({
    purpose: '',
    dataSize: '',
    compute: '',
    priority: ''
  });

  const questions = [
    {
      id: 'purpose',
      title: 'What is your primary goal?',
      description: 'This helps us understand what you want to achieve',
      options: [
        {
          value: 'experiment',
          label: 'Quick Experimentation',
          desc: 'Test ideas and iterate fast'
        },
        {
          value: 'learn',
          label: 'Learning & Education',
          desc: 'Understand how GPT models work'
        },
        {
          value: 'production',
          label: 'Production Quality',
          desc: 'Best possible model performance'
        },
        {
          value: 'research',
          label: 'Research Project',
          desc: 'Academic or professional research'
        }
      ]
    },
    {
      id: 'dataSize',
      title: 'How much training data do you have?',
      description: 'Larger datasets can support bigger models',
      options: [
        {
          value: 'tiny',
          label: 'Very Small',
          desc: 'Less than 10KB (a few paragraphs)'
        },
        {
          value: 'small',
          label: 'Small',
          desc: '10KB - 100KB (few pages)'
        },
        {
          value: 'medium',
          label: 'Medium',
          desc: '100KB - 1MB (book-sized)'
        },
        {
          value: 'large',
          label: 'Large',
          desc: 'Over 1MB (multiple books)'
        }
      ]
    },
    {
      id: 'compute',
      title: 'What are your compute resources?',
      description: 'Be honest about your hardware capabilities',
      options: [
        {
          value: 'low',
          label: 'Limited',
          desc: 'CPU only or older GPU'
        },
        {
          value: 'medium',
          label: 'Moderate',
          desc: 'Modern GPU (GTX/RTX series)'
        },
        {
          value: 'high',
          label: 'Powerful',
          desc: 'High-end GPU or multiple GPUs'
        }
      ]
    },
    {
      id: 'priority',
      title: 'What matters most to you?',
      description: 'Choose your primary concern',
      options: [
        {
          value: 'speed',
          label: 'Training Speed',
          desc: 'Get results quickly'
        },
        {
          value: 'quality',
          label: 'Output Quality',
          desc: 'Best possible generation'
        },
        {
          value: 'balance',
          label: 'Balanced',
          desc: 'Good mix of speed and quality'
        }
      ]
    }
  ];

  const currentQuestion = questions[step];

  const selectOption = (value) => {
    setAnswers(prev => ({
      ...prev,
      [currentQuestion.id]: value
    }));
  };

  const nextStep = () => {
    if (step < questions.length - 1) {
      setStep(step + 1);
    } else {
      // Final step - generate recommendation
      const config = generateRecommendation(answers);
      onComplete(config);
    }
  };

  const prevStep = () => {
    if (step > 0) {
      setStep(step - 1);
    }
  };

  const isCurrentAnswered = answers[currentQuestion.id] !== '';

  return (
    <div className="wizard-overlay">
      <div className="wizard-modal">
        <div className="wizard-header">
          <div className="wizard-icon">
            <Wand2 size={32} />
          </div>
          <h2>Setup Wizard</h2>
          <p>Let's find the perfect configuration for your needs</p>
          <button className="wizard-close" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div className="wizard-progress">
          {questions.map((_, idx) => (
            <div
              key={idx}
              className={`progress-step ${idx <= step ? 'active' : ''} ${idx < step ? 'completed' : ''}`}
            >
              {idx < step ? <Check size={16} /> : idx + 1}
            </div>
          ))}
        </div>

        <div className="wizard-content">
          <div className="wizard-question">
            <div className="question-number">Question {step + 1} of {questions.length}</div>
            <h3>{currentQuestion.title}</h3>
            <p className="question-desc">{currentQuestion.description}</p>
          </div>

          <div className="wizard-options">
            {currentQuestion.options.map(option => (
              <button
                key={option.value}
                className={`wizard-option ${answers[currentQuestion.id] === option.value ? 'selected' : ''}`}
                onClick={() => selectOption(option.value)}
              >
                <div className="option-header">
                  <div className="option-radio">
                    {answers[currentQuestion.id] === option.value && <div className="radio-dot" />}
                  </div>
                  <div className="option-content">
                    <div className="option-label">{option.label}</div>
                    <div className="option-desc">{option.desc}</div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="wizard-footer">
          <button
            className="btn-secondary"
            onClick={prevStep}
            disabled={step === 0}
          >
            <ChevronLeft size={18} />
            Previous
          </button>
          <button
            className="btn-primary"
            onClick={nextStep}
            disabled={!isCurrentAnswered}
          >
            {step === questions.length - 1 ? 'Get Recommendation' : 'Next'}
            <ChevronRight size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}

// Generate configuration recommendation based on answers
function generateRecommendation(answers) {
  const { purpose, dataSize, compute, priority } = answers;

  // Base configurations
  const configs = {
    minimal: {
      epochs: 3,
      batch_size: 64,
      learning_rate: 0.0005,
      seq_length: 32,
      d_model: 64,
      num_heads: 2,
      num_layers: 2,
      d_ff: 256
    },
    tiny: {
      epochs: 5,
      batch_size: 32,
      learning_rate: 0.0003,
      seq_length: 64,
      d_model: 128,
      num_heads: 4,
      num_layers: 4,
      d_ff: 512
    },
    small: {
      epochs: 10,
      batch_size: 32,
      learning_rate: 0.0003,
      seq_length: 128,
      d_model: 256,
      num_heads: 8,
      num_layers: 6,
      d_ff: 1024
    },
    medium: {
      epochs: 20,
      batch_size: 16,
      learning_rate: 0.0002,
      seq_length: 256,
      d_model: 512,
      num_heads: 8,
      num_layers: 8,
      d_ff: 2048
    },
    large: {
      epochs: 30,
      batch_size: 8,
      learning_rate: 0.00015,
      seq_length: 512,
      d_model: 768,
      num_heads: 12,
      num_layers: 12,
      d_ff: 3072
    }
  };

  // Decision logic
  let selectedConfig = 'small'; // default

  // Purpose-based adjustments
  if (purpose === 'experiment') {
    selectedConfig = 'minimal';
  } else if (purpose === 'learn') {
    selectedConfig = 'tiny';
  } else if (purpose === 'production' || purpose === 'research') {
    if (dataSize === 'large' && compute === 'high') {
      selectedConfig = 'large';
    } else if (dataSize === 'medium' || compute === 'medium') {
      selectedConfig = 'medium';
    } else {
      selectedConfig = 'small';
    }
  }

  // Compute constraints
  if (compute === 'low') {
    if (selectedConfig === 'large' || selectedConfig === 'medium') {
      selectedConfig = 'small';
    }
    if (selectedConfig === 'small') {
      selectedConfig = 'tiny';
    }
  }

  // Data size adjustments
  if (dataSize === 'tiny') {
    if (selectedConfig !== 'minimal') {
      selectedConfig = 'tiny';
    }
  } else if (dataSize === 'large' && compute !== 'low') {
    if (selectedConfig === 'small' || selectedConfig === 'tiny') {
      selectedConfig = 'medium';
    }
  }

  // Priority adjustments
  let config = { ...configs[selectedConfig] };

  if (priority === 'speed') {
    config.epochs = Math.max(3, Math.floor(config.epochs * 0.5));
    config.batch_size = Math.min(64, config.batch_size * 2);
  } else if (priority === 'quality') {
    config.epochs = Math.floor(config.epochs * 1.5);
    config.learning_rate = config.learning_rate * 0.8;
  }

  return config;
}

export default SetupWizard;
