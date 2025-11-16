# 🤖 AI Training Platform

**Train transformer models from scratch with a beautiful, modern web interface.**

![Platform](https://img.shields.io/badge/Status-Ready-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![React](https://img.shields.io/badge/React-18+-61dafb)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)

A complete, educational platform for training GPT-style transformer models from scratch. Perfect for learning how modern AI systems work!

## ✨ Features

- 🧠 **Full GPT-style Transformer** - Complete implementation with multi-head attention
- 📊 **Real-time Visualization** - Watch your model learn with live charts and metrics
- 💬 **Interactive Chat** - Test your model by chatting with it like ChatGPT
- 🗄️ **Flexible Dataset Management** - Load data from files, URLs, or paste directly
- ⚙️ **Configurable Architecture** - Adjust model size, layers, and training parameters
- 🎨 **Modern UI** - Beautiful, responsive web interface built with React
- 🔌 **WebSocket Updates** - Real-time training progress without refreshing

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- ~2GB free disk space

### Installation

1. **Clone or download this repository**

2. **Install backend dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

3. **Install frontend dependencies:**
```bash
cd frontend
npm install
```

### Running the Application

1. **Start the backend server:**
```bash
cd backend
python server.py
```
Server runs at `http://localhost:8000`

2. **In a new terminal, start the frontend:**
```bash
cd frontend
npm run dev
```
Web interface opens at `http://localhost:2121`

3. **Open your browser and go to:** `http://localhost:2121`

## 🎓 Your First Training Session

1. **Add Data**
   - Go to the "Dataset" tab
   - Click "Add Sample Data" for a quick start
   - Or paste your own text, or load from a URL

2. **Configure Model**
   - Go to "Configuration" tab
   - Click "Tiny Model" preset for fast training
   - Or adjust parameters manually

3. **Start Training**
   - Click "Start Training" button (top right)
   - Switch to "Training" tab to watch progress
   - See loss decrease and metrics update in real-time

4. **Chat with Your AI**
   - Once training starts, go to "Chat" tab
   - Type a message and click "Send"
   - Experiment with temperature and length settings

## 📁 Project Structure

```
GPT/
├── backend/
│   ├── model.py          # Transformer architecture
│   ├── trainer.py        # Training pipeline
│   ├── dataset.py        # Data loading & preprocessing
│   ├── server.py         # FastAPI server
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx                      # Main application
│   │   ├── components/
│   │   │   ├── DatasetManager.jsx       # Dataset interface
│   │   │   ├── TrainingDashboard.jsx    # Training visualization
│   │   │   ├── ChatInterface.jsx        # Chat with model
│   │   │   └── ConfigPanel.jsx          # Configuration
│   │   └── index.css                    # Styles
│   ├── package.json      # Node dependencies
│   └── vite.config.js    # Vite configuration
├── INSTRUCTIONS.md       # Comprehensive manual
└── README.md            # This file
```

## 🎯 Key Concepts

### The Transformer Architecture

Our model implements a simplified GPT-style transformer:

- **Token Embeddings**: Convert text to numerical vectors
- **Positional Encoding**: Add position information
- **Multi-Head Attention**: Let the model focus on relevant parts of input
- **Feed-Forward Networks**: Process and transform information
- **Layer Normalization**: Stabilize training
- **Causal Masking**: Prevent looking at future tokens

### Training Process

1. **Forward Pass**: Input text → Model → Predicted next character
2. **Loss Calculation**: Compare prediction to actual next character
3. **Backpropagation**: Calculate gradients
4. **Optimizer Step**: Update weights to reduce loss
5. **Repeat**: Thousands of iterations until convergence

### Metrics Explained

- **Loss**: How wrong the predictions are (lower is better)
- **Perplexity**: Model uncertainty (lower is better)
- **Learning Rate**: Size of weight updates
- **Tokens/sec**: Training speed

## ⚙️ Configuration Options

### Model Architecture

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| d_model | Embedding dimension | 128, 256, 512 |
| num_heads | Attention heads | 4, 8, 16 |
| num_layers | Transformer blocks | 4, 6, 12 |
| d_ff | Feed-forward dimension | 512, 1024, 2048 |
| seq_length | Maximum context | 128, 256, 512 |

### Training Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| epochs | Training iterations | 5, 10, 20 |
| batch_size | Samples per step | 16, 32, 64 |
| learning_rate | Update step size | 0.0001, 0.0003, 0.001 |

### Presets

- **Tiny**: ~1M params, trains in minutes
- **Small**: ~10M params, balanced performance
- **Medium**: ~50M params, better quality

## 🔧 Advanced Usage

### Using Your Own Data

**From Files:**
```python
# In Python
from dataset import DatasetManager
manager = DatasetManager()
manager.add_file('path/to/your/file.txt')
```

**From URLs:**
Just paste any URL in the web interface, and it will extract the text automatically.

**Best Practices:**
- Use at least 10,000 characters
- Clean, well-formatted text works best
- Multiple diverse sources improve quality
- Match the style you want the model to generate

### Training Tips

1. **Start Small**: Use "Tiny" preset first to verify everything works
2. **Monitor Loss**: Should steadily decrease
3. **Be Patient**: Good models need time (30-60 minutes)
4. **Experiment**: Try different architectures and data
5. **Save Checkpoints**: Models auto-save after each epoch

### GPU Acceleration

If you have an NVIDIA GPU:
```bash
# Check if PyTorch detects your GPU
python -c "import torch; print(torch.cuda.is_available())"
```

The platform automatically uses GPU if available. Expect 10-50x speedup!

## 📊 API Endpoints

The backend provides a REST API:

- `POST /dataset/add-text` - Add training text
- `POST /dataset/add-url` - Fetch text from URL
- `POST /training/start` - Begin training
- `POST /training/stop` - Stop training
- `GET /training/status` - Get current status
- `POST /chat` - Generate text from model
- `WS /ws` - WebSocket for real-time updates

Full API docs: `http://localhost:8000/docs`

## 🎨 Web Interface

### Dataset Tab
- Add custom text
- Load from URLs
- Quick sample data
- View dataset statistics

### Configuration Tab
- Adjust model architecture
- Set training hyperparameters
- Use quick presets
- See parameter estimates

### Training Tab
- Real-time loss/perplexity charts
- Live training metrics
- Progress tracking
- Performance statistics

### Chat Tab
- Interactive conversation
- Adjustable temperature
- Variable response length
- Message history

## 🐛 Troubleshooting

### "Module not found"
```bash
pip install -r backend/requirements.txt --upgrade
```

### "Cannot connect to backend"
- Ensure backend server is running on port 8000
- Check for firewall issues
- Verify `http://localhost:8000` opens in browser

### "Training is very slow"
- Use smaller model (Tiny preset)
- Reduce batch size
- Training on CPU is slower than GPU
- Close other programs to free resources

### "Loss is NaN"
- Learning rate too high, try 0.0001
- Restart training with lower learning rate

### "Model outputs gibberish"
- Train for more epochs
- Add more/better training data
- Use larger model architecture
- Lower temperature in chat

For more help, see [INSTRUCTIONS.md](INSTRUCTIONS.md#troubleshooting)

## 📚 Learn More

### Understanding Transformers
- Read the code comments in `backend/model.py`
- Study attention mechanism in `model.py:23-71`
- Review training loop in `trainer.py:112-174`

### Key Resources
- Original Paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

### Full Documentation
See [INSTRUCTIONS.md](INSTRUCTIONS.md) for comprehensive guide including:
- Detailed architecture explanation
- Line-by-line code walkthrough
- Advanced training techniques
- Hyperparameter tuning guide
- Extension ideas

## 🚧 Future Enhancements

Potential improvements:
- [ ] Word-level or BPE tokenization
- [ ] Validation dataset support
- [ ] Model comparison tools
- [ ] Checkpoint management UI
- [ ] Distributed training support
- [ ] Pre-trained model zoo
- [ ] Fine-tuning interface
- [ ] Export to ONNX/TorchScript

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment and modify the code
- Add new features
- Improve documentation
- Share your trained models

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - UI library
- [Recharts](https://recharts.org/) - Charting library
- [Vite](https://vitejs.dev/) - Frontend build tool

Inspired by the original transformer architecture from "Attention Is All You Need" and GPT models.

---

## 🎉 Get Started Now!

```bash
# Terminal 1: Start backend
cd backend && python server.py

# Terminal 2: Start frontend
cd frontend && npm run dev

# Open browser
# http://localhost:2121
```

**Happy training! 🚀🤖**

---

*For detailed instructions, see [INSTRUCTIONS.md](INSTRUCTIONS.md)*
