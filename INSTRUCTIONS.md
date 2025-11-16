# AI Training Platform - Complete Instructions

Welcome to your AI Training Platform! This guide will teach you everything you need to know about training transformer models from scratch.

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Understanding the Architecture](#understanding-the-architecture)
4. [Detailed Setup](#detailed-setup)
5. [Using the Platform](#using-the-platform)
6. [Training Your First Model](#training-your-first-model)
7. [Understanding the Results](#understanding-the-results)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is This Platform?

This platform allows you to:
- **Train transformer models from scratch** - Learn how AI actually works by building it yourself
- **Visualize training in real-time** - Watch your model learn with live charts and metrics
- **Chat with your trained models** - Test your AI by having conversations with it
- **Experiment with different architectures** - Adjust model size, layers, and hyperparameters

### What You'll Learn

By using this platform, you'll understand:
- How transformer architectures work (the technology behind ChatGPT, Claude, etc.)
- What "training" actually means in machine learning
- How neural networks learn from data
- The importance of hyperparameters and architecture choices

### Prerequisites

- **Python 3.8+** installed on your computer
- **Node.js 16+** for the web interface
- Basic understanding of command line/terminal
- ~2GB free disk space
- (Optional) NVIDIA GPU for faster training

---

## Quick Start

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Start the Backend Server

```bash
cd backend
python server.py
```

The API server will start at `http://localhost:8000`

### 4. Start the Frontend (in a new terminal)

```bash
cd frontend
npm run dev
```

The web interface will open at `http://localhost:2121`

### 5. Train Your First Model

1. Open `http://localhost:2121` in your browser
2. Go to the **Dataset** tab
3. Click **"Add Sample Data"** to load example text
4. Go to the **Configuration** tab and click **"Tiny Model"** preset
5. Click **"Start Training"** in the top right
6. Watch it train in the **Training** tab!

---

## Understanding the Architecture

### The Transformer Model

Our transformer model consists of several key components:

#### 1. **Token Embeddings** (`backend/model.py:142-143`)
- Converts text characters into numerical vectors
- Each character gets a unique vector representation
- Think of it as teaching the model a "language" of numbers

#### 2. **Positional Encoding** (`backend/model.py:146-147`)
- Adds position information to each token
- Helps the model understand word order
- "The dog bit the man" vs "The man bit the dog" = different meanings!

#### 3. **Self-Attention Mechanism** (`backend/model.py:23-71`)
- The "brain" of the transformer
- Allows the model to focus on relevant parts of the input
- When processing "it", attention helps identify what "it" refers to

#### 4. **Feed-Forward Networks** (`backend/model.py:74-93`)
- Processes information from attention layers
- Adds non-linearity and complexity
- Helps the model learn complex patterns

#### 5. **Transformer Blocks** (`backend/model.py:96-122`)
- Combines attention + feed-forward layers
- Stacked multiple times for deeper understanding
- More blocks = smarter model (but slower training)

#### 6. **Output Layer** (`backend/model.py:157`)
- Predicts the next character/token
- Outputs probabilities for each possible next character

### How Training Works

1. **Forward Pass**: Input text → Model → Predicted next character
2. **Loss Calculation**: Compare prediction to actual next character
3. **Backward Pass**: Calculate how to adjust weights to reduce loss
4. **Optimizer Step**: Update model weights slightly
5. **Repeat** thousands of times!

The model gradually learns patterns in your training data.

---

## Detailed Setup

### Backend Setup

The backend is built with:
- **PyTorch**: Deep learning framework
- **FastAPI**: Modern web framework for APIs
- **WebSocket**: Real-time communication with frontend

**Installation:**

```bash
cd backend
pip install -r requirements.txt
```

**What gets installed:**
- `torch`: The PyTorch deep learning library
- `fastapi`: API framework
- `uvicorn`: ASGI server to run FastAPI
- `websockets`: Real-time communication
- `requests`: For fetching data from URLs
- `numpy`: Numerical computations

### Frontend Setup

The frontend is built with:
- **React**: UI library
- **Vite**: Fast build tool
- **Recharts**: Beautiful charts for training visualization

**Installation:**

```bash
cd frontend
npm install
```

**What gets installed:**
- `react` & `react-dom`: UI framework
- `recharts`: Charting library
- `lucide-react`: Beautiful icons
- `vite`: Development server and build tool

---

## Using the Platform

### 1. Dataset Management Tab

This is where you add training data for your model.

#### Adding Custom Text
1. Click on the **Dataset** tab
2. Paste or type text in the "Add Custom Text" box
3. Click **"Add Text"**

**Best practices:**
- Use text in a consistent style
- More data = better model (aim for 10,000+ characters)
- Clean, well-formatted text works best

#### Loading from URLs
1. Enter a URL (e.g., a Wikipedia article)
2. Click **"Fetch URL"**
3. The system extracts text automatically

**Tips:**
- News articles, blog posts, and Wikipedia work well
- Avoid URLs with lots of ads or navigation
- Some websites may block automated access

#### Sample Data
Click **"Add Sample Data"** to load a small example about transformers. Perfect for testing!

### 2. Configuration Tab

Adjust your model's architecture and training parameters.

#### Training Parameters

**Epochs** (1-100)
- How many times to go through your entire dataset
- More epochs = more learning (but can overfit)
- Start with 5-10 for experiments

**Batch Size** (1-128)
- How many samples to process at once
- Larger = faster training (if you have enough memory)
- Smaller = more stable training
- Typical values: 16, 32, 64

**Learning Rate** (0.00001-0.01)
- How big of steps to take when updating weights
- Too high = unstable training, won't converge
- Too low = very slow learning
- Default 0.0003 works well for most cases

**Sequence Length** (32-1024)
- Maximum context the model can see at once
- Longer = more context but slower training
- Start with 128, increase if needed

#### Model Architecture

**Model Dimension (d_model)** (64-1024)
- Size of embedding and hidden layers
- Larger = more capacity but slower
- Common values: 128, 256, 512

**Number of Heads** (1-16)
- Parallel attention mechanisms
- More heads = can focus on different aspects
- Must divide evenly into d_model
- Common values: 4, 8, 16

**Number of Layers** (1-24)
- How many transformer blocks to stack
- More layers = deeper understanding
- But also slower training and more memory
- Start with 4-6 layers

**Feed-Forward Dimension** (256-4096)
- Size of intermediate FFN layer
- Typically 4x the model dimension
- Larger = more capacity

#### Quick Presets

- **Tiny Model**: ~1M parameters, trains in minutes, good for testing
- **Small Model**: ~10M parameters, balanced performance
- **Medium Model**: ~50M parameters, better quality, needs more time/resources

### 3. Training Tab

Monitor your model's training progress in real-time.

#### Metrics Explained

**Loss**
- Measures how wrong the model's predictions are
- Lower is better
- Should decrease over time
- If it's not decreasing, something's wrong

**Perplexity**
- Another measure of model quality (exp(loss))
- Lower is better
- Roughly: "how confused is the model?"
- Good models: 10-50, Great models: < 10

**Learning Rate**
- Shows current learning rate (changes with scheduler)
- Decreases over training for fine-tuning

**Tokens/sec**
- Training speed
- Higher = faster training
- Depends on hardware and model size

#### Charts

**Loss Chart**
- Should trend downward
- Some fluctuation is normal
- Flat line = model stopped learning

**Perplexity Chart**
- Should decrease over time
- Mirrors the loss chart

**Progress Info**
- Current epoch and batch
- Estimated time remaining

### 4. Chat Tab

Interact with your trained model!

#### Settings

**Temperature** (0.1-2.0)
- Controls randomness of generation
- Low (0.5): More predictable, repetitive
- Medium (0.8): Balanced
- High (1.5+): More creative, sometimes nonsensical

**Max Length** (20-500)
- Maximum tokens to generate
- Longer = more text but slower
- Start with 100

#### Tips for Better Responses

1. **Train first!** The model needs training before it can chat
2. **Be patient with small models** - They won't be as coherent as ChatGPT
3. **Experiment with temperature** - Different values give different styles
4. **Provide context** - Start prompts with relevant information
5. **Keep expectations realistic** - This is a small model trained on limited data

---

## Training Your First Model

Let's walk through a complete training session step by step.

### Step 1: Prepare Your Data

**Option A: Use Sample Data (Fastest)**
1. Go to **Dataset** tab
2. Click **"Add Sample Data"**
3. Done!

**Option B: Use Custom Text (Recommended)**
1. Find interesting text (stories, articles, code, etc.)
2. Copy and paste into the text box
3. Add multiple sources for variety
4. Aim for at least 10,000 characters

**Option C: Load from URL**
1. Find a good article or page
2. Paste URL and click **"Fetch URL"**
3. Repeat with multiple URLs for better results

### Step 2: Configure Your Model

1. Go to **Configuration** tab
2. For your first model, click **"Tiny Model"** preset
3. This ensures fast training so you can see results quickly

### Step 3: Start Training

1. Click **"Start Training"** (top right)
2. Switch to **Training** tab
3. Watch the metrics!

**What to expect:**
- Loss should start high (5-10) and decrease
- First epoch might take a few minutes
- You'll see real-time charts updating

### Step 4: Monitor Progress

**Healthy training looks like:**
- Loss steadily decreasing
- Perplexity dropping
- Smooth curves (some noise is ok)

**Problems:**
- Loss stuck or increasing → Try lower learning rate
- Loss = NaN → Learning rate too high, restart
- Very slow progress → Increase learning rate slightly

### Step 5: Test Your Model

1. Once training completes (or even during!), go to **Chat** tab
2. Type a prompt related to your training data
3. Adjust temperature and max length
4. Click **"Send"**

**Example prompts:**
- If trained on articles: "The main advantage of"
- If trained on stories: "Once upon a time"
- General: "Hello, how are you?"

---

## Understanding the Results

### Reading the Metrics

#### Loss Values

| Loss | Model Quality |
|------|--------------|
| > 5.0 | Just started training |
| 2.0-5.0 | Learning basic patterns |
| 1.0-2.0 | Good progress, coherent output |
| 0.5-1.0 | Excellent, near-perfect on training data |
| < 0.5 | Likely overfitting |

#### Perplexity Values

| Perplexity | What It Means |
|------------|---------------|
| > 100 | Very confused, just started |
| 20-100 | Learning patterns |
| 5-20 | Good quality |
| < 5 | Excellent quality |

### Generation Quality

**Signs of a good model:**
- Outputs relevant to prompt
- Maintains consistent style
- Forms complete words/phrases
- Some logical coherence

**Signs of underfitting:**
- Random characters
- Ignores prompt completely
- No recognizable patterns
- High loss/perplexity

**Signs of overfitting:**
- Repeats training data exactly
- Can't generalize to new prompts
- Low loss but poor actual performance
- Needs more diverse training data

---

## Advanced Topics

### Optimizing Training Speed

#### Use GPU Acceleration
If you have an NVIDIA GPU:
```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"
```
The code automatically uses GPU if available!

#### Batch Size Tuning
- Increase batch size until you hit memory limits
- Larger batches = faster training
- On GPU: Try 64, 128, or even 256
- On CPU: Stick to 16-32

#### Model Size
- Smaller models train much faster
- Start with Tiny, scale up as needed
- Parameter count vs speed:
  - 1M: Very fast, limited quality
  - 10M: Good balance
  - 50M+: High quality, slow training

### Improving Model Quality

#### More/Better Data
- Quality > Quantity
- Diverse sources help generalization
- Clean, well-formatted text works best
- 50,000+ characters recommended for good results

#### Longer Training
- Increase epochs (20-50 for small datasets)
- Let loss plateau before stopping
- Watch for overfitting (perfect training, poor testing)

#### Bigger Models
- Use "Small" or "Medium" preset
- More layers capture deeper patterns
- Larger d_model increases capacity
- Needs more data to avoid overfitting

#### Hyperparameter Tuning
Try different values:
- Learning rate: 0.0001, 0.0003, 0.001
- Sequence length: Match your typical input length
- Adjust based on results

### Saving and Loading Models

Models are automatically saved to `./checkpoints/` directory:
- After each epoch: `checkpoint_epoch_N.pt`
- Final model: `final_model.pt`
- If interrupted: `interrupted_checkpoint.pt`

**To load a saved model:**
1. Use the `/model/load` API endpoint
2. Or modify `server.py` to load on startup

### Understanding the Code

#### Model Architecture (`backend/model.py`)
- Line 23-71: Multi-head attention implementation
- Line 74-93: Feed-forward network
- Line 96-122: Transformer block combining both
- Line 125-228: Complete GPT model
- Line 230-260: Text generation logic

#### Training Loop (`backend/trainer.py`)
- Line 65-104: Single training step
- Line 112-174: Full epoch training
- Line 176-224: Complete training loop with checkpointing

#### Dataset Loading (`backend/dataset.py`)
- Line 21-52: Character tokenizer
- Line 55-87: Dataset class for batching
- Line 90-209: High-level dataset manager

---

## Troubleshooting

### Backend Issues

**"Module not found" errors**
```bash
cd backend
pip install -r requirements.txt --upgrade
```

**"CUDA out of memory"**
- Reduce batch size
- Use smaller model
- Close other GPU applications

**"Address already in use"**
- Another process is using port 8000
- Kill it or change port in `server.py`

**Training is very slow**
- CPU training is slow, use smaller models
- Reduce batch size if swapping to disk
- Close other programs

### Frontend Issues

**"Cannot connect to backend"**
- Make sure backend server is running
- Check `http://localhost:8000` opens in browser
- Verify API_URL in `App.jsx` is correct

**"npm install" fails**
- Update Node.js to version 16+
- Try: `npm cache clean --force`
- Delete `node_modules` and try again

**Charts not displaying**
- Wait for training to start
- Check browser console for errors
- Verify WebSocket connection is working

### Training Issues

**Loss is NaN**
- Learning rate too high, try 0.0001
- Batch size too large
- Data has issues (check for invalid characters)

**Loss not decreasing**
- Learning rate too low, try 0.001
- Model too small for data complexity
- Not enough training data
- Need more epochs

**Model outputs gibberish**
- Not trained enough, continue training
- Temperature too high, lower it
- Model too small, use bigger architecture
- Training data was too random/noisy

**Training crashes**
- Out of memory: reduce batch size or model size
- Bad data: check for encoding issues
- Update PyTorch: `pip install torch --upgrade`

### Common Questions

**Q: How long should training take?**
A: Depends on hardware and model size:
- Tiny model on CPU: 5-10 minutes
- Small model on CPU: 30-60 minutes
- Medium model on GPU: 30-90 minutes

**Q: How much data do I need?**
A: Minimum 5,000 characters, recommended 50,000+. More is better!

**Q: Why isn't my model as good as ChatGPT?**
A: ChatGPT has:
- Billions of parameters (yours has millions)
- Trained on hundreds of GBs of data
- Months of training on supercomputers
- Advanced fine-tuning techniques

This platform is for learning how it works, not competing with commercial models!

**Q: Can I train on my own language/domain?**
A: Absolutely! Just add data in your target language or domain. The character-level tokenizer works with any text.

**Q: How do I deploy my model?**
A: The chat interface is already functional! For production:
- Save your checkpoint
- Load it in `server.py` on startup
- Deploy backend and frontend to a server
- Use HTTPS and proper authentication

---

## Next Steps

### Experiment!
- Try different types of training data
- Adjust hyperparameters and see what happens
- Compare tiny vs small vs medium models
- Train multiple models and compare results

### Learn More
- Read the code comments in `backend/`
- Study the transformer architecture paper: "Attention Is All You Need"
- Explore PyTorch documentation
- Try implementing new features

### Extend the Platform
Ideas for enhancements:
- Add word-level or BPE tokenization
- Implement model comparison tools
- Add validation datasets
- Create data augmentation
- Build a model zoo with pre-trained models
- Add fine-tuning capabilities
- Implement beam search for generation

---

## Resources

### Papers
- "Attention Is All You Need" - Original transformer paper
- "Language Models are Unsupervised Multitask Learners" - GPT-2 paper

### Tutorials
- PyTorch tutorials: pytorch.org/tutorials
- FastAPI documentation: fastapi.tiangolo.com
- React documentation: react.dev

### Communities
- r/MachineLearning on Reddit
- PyTorch forums
- Hugging Face forums

---

## Credits

This platform was built to teach the fundamentals of transformer models and neural network training. The architecture is inspired by GPT but simplified for educational purposes.

**Technologies used:**
- PyTorch for deep learning
- FastAPI for the backend API
- React for the frontend UI
- Recharts for visualization

---

## License & Usage

Feel free to use this platform for:
- Learning and education
- Research and experimentation
- Building your own AI projects
- Teaching others about transformers

Happy training! 🚀🤖
